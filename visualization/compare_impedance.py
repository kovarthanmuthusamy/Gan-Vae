"""
Compare Generated vs Real Impedance Profiles
=============================================
This script plots multiple generated vs real impedance profile comparisons
in subplots for easy visual analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION - Update these paths for your comparisons
# ============================================================

# Define input directory and output path
input_dir = Path("temp_visuals")
OUTPUT_PATH = input_dir / "generated_vs_real_impedance_profile.png"

# 2. Automate the list creation
# This creates a list for indices 0, 1, 2, 3, and 4
COMPARISONS = [
    {
        "generated": input_dir / f"data_sample_{i}" / "impedance_integrated.npy",
        "derivative": input_dir / f"data_sample_{i}" / "impedance_derivative.npy",
        "real": input_dir / f"data_sample_{i}" / "Real_Imp.csv",
        "label": f"data_sample_{i}"
    }
    for i in range(0, 5)  # Generate comparisons for samples 0 to 4
]

# Target impedance path
TARGET_IMPEDANCE_PATH = Path("configs/target_impedance.npy")

# Frequency data path
FREQUENCY_PATH = Path("configs/Frequency_data_hz.npy")


# ============================================================


def load_impedance(path):
    """Load impedance from CSV or NPY file."""
    path = Path(path)
    if path.suffix == '.csv':
        import pandas as pd
        
        # Try to detect the file format by reading first few lines
        with open(path, 'r') as f:
            first_lines = [f.readline() for _ in range(10)]
        
        # Check if it's a Zuken format file (starts with #)
        if first_lines[0].startswith('#'):
            # Find the line with actual data headers (1_f, 1_Z)
            skip_rows = 0
            for i, line in enumerate(first_lines):
                if line.strip().startswith('1_f') or line.strip().startswith('(Hz)'):
                    skip_rows = i
                    break
                if '(Hz)' in line:
                    skip_rows = i + 1
                    break
            
            # Skip header rows and unit row
            df = pd.read_csv(path, skiprows=skip_rows, header=0)
            # Skip the units row if present
            if df.iloc[0, 0] == '(Hz)' or '(Hz)' in str(df.iloc[0, 0]):
                df = df.iloc[1:]
            
            # Get impedance column (second column, typically "1_Z")
            data = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        else:
            # Standard CSV format
            df = pd.read_csv(path)
            # Try common column names
            for col in ['Impedance_Ohms', 'Impedance', 'impedance', 'value', '1_Z']:
                if col in df.columns:
                    data = df[col].values
                    break
            else:
                # Use second column if no known column name
                data = df.iloc[:, 1].values
    elif path.suffix == '.npy':
        # impedance_profile.npy is saved in log scale from inference; convert to Ohm
        data = np.exp(np.load(path).flatten())
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    return data


def plot_comparison(frequency, target_impedance, comparisons_data, output_path):
    """
    Plot multiple generated vs real impedance profiles in subplots.
    
    comparisons_data: list of dicts with 'generated', 'real', 'label' keys
    """
    n_plots = len(comparisons_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots))
    
    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    
    for idx, (ax, data) in enumerate(zip(axes, comparisons_data)):
        generated = data['generated']
        real = data['real']
        
        # Plot target impedance (dashed red)
        ax.loglog(
            frequency, 
            target_impedance.flatten(), 
            linestyle='--', 
            linewidth=1, 
            color='red', 
            label='Target'
        )
        
        # Plot real impedance (solid blue)
        ax.loglog(
            frequency, 
            real, 
            linestyle='-', 
            linewidth=1, 
            color='blue', 
            label='Real'
        )
        
        # Plot generated impedance (solid green)
        ax.loglog(
            frequency, 
            generated, 
            linestyle='-', 
            linewidth=1, 
            color='green', 
            label='Generated (integrated)'
        )
        
        # Plot derivative on twin y-axis (dotted orange)
        derivative = data.get('derivative')
        if derivative is not None:
            ax2 = ax.twinx()
            ax2.semilogx(
                frequency,
                derivative,
                linestyle=':',
                linewidth=1.2,
                color='darkorange',
                label='Derivative (z-score)'
            )
            ax2.axhline(0, color='darkorange', linewidth=0.6, linestyle=':', alpha=0.4)
            ax2.set_ylabel('First Derivative (z-score)', fontsize=11, color='darkorange')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='best')
        else:
            ax.legend(fontsize=10, loc='best')
        
        # Formatting
        ax.set_ylim(1e-3, 1e2)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Impedance (Ohm)", fontsize=12)
        ax.set_title(f"{data['label']}: Generated vs Real", fontsize=14)
        ax.grid(True, which="both", linestyle='--', alpha=0.7)
    
    plt.suptitle("Impedance Profile Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison plot: {output_path}")


def main():
    # Get repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    # Load frequency data
    frequency = np.load(repo_root / FREQUENCY_PATH)
    print(f"Loaded frequency data: {frequency.shape}")
    
    # Load target impedance
    target_impedance = np.load(repo_root / TARGET_IMPEDANCE_PATH)
    print(f"Loaded target impedance: {target_impedance.shape}")
    
    # Load all comparisons
    comparisons_data = []
    for comp in COMPARISONS:
        # Load generated impedance (log scale -> exp -> Ohm)
        generated = load_impedance(repo_root / comp["generated"])
        print(f"Loaded generated impedance for {comp['label']}: {generated.shape}")
        print(f"  Generated range: [{generated.min():.6e}, {generated.max():.6e}]")
        
        # Load derivative (z-score, raw)
        deriv_path = repo_root / comp["derivative"]
        derivative = np.load(deriv_path).flatten() if deriv_path.exists() else None
        
        # Load real impedance (already in Ohm scale from CSV)
        real = load_impedance(repo_root / comp["real"])
        print(f"Loaded real impedance for {comp['label']}: {real.shape}")
        print(f"  Real range: [{real.min():.6e}, {real.max():.6e}]")
        
        comparisons_data.append({
            "generated": generated,
            "derivative": derivative,
            "real": real,
            "label": comp["label"]
        })
    
    # Plot comparison
    plot_comparison(
        frequency=frequency,
        target_impedance=target_impedance,
        comparisons_data=comparisons_data,
        output_path=repo_root / OUTPUT_PATH
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("IMPEDANCE STATISTICS")
    print("=" * 60)
    for data in comparisons_data:
        print(f"\n{data['label']}:")
        print(f"  Generated - Min: {data['generated'].min():.6f}, Max: {data['generated'].max():.6f}, Mean: {data['generated'].mean():.6f}")
        print(f"  Real      - Min: {data['real'].min():.6f}, Max: {data['real'].max():.6f}, Mean: {data['real'].mean():.6f}")


if __name__ == "__main__":
    main()
