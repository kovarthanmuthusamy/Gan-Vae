"""Impedance file reading and processing."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EXPECTED_IMP_LENGTH = 231

def _read_csv(filepath, cols=None, **kwargs):
    """Generic CSV reader."""
    try:
        df = pd.read_csv(filepath, **kwargs)
        return df[cols] if cols else df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def read_impedance_file(filepath):
    """Reads impedance values from .csv file.
    
    Args:
        filepath: Path to impedance CSV file
        
    Returns:
        numpy array of impedance values (EXPECTED_IMP_LENGTH, ) or None if error
    """
    try:
        data = pd.read_csv(filepath, delimiter=";", skiprows=4, usecols=[1], dtype=np.float32, engine="c")
        imp = data.values.flatten().astype(np.float32)
        # Validate length - must be exactly 231
        if len(imp) != EXPECTED_IMP_LENGTH:
            return None
        return imp
    except Exception:
        return None

def visualize_impedance(impedance_file, output_path=None, show=True):
    """Visualize impedance profile from saved numpy file.
    
    Args:
        impedance_file: Path to impedance .npy file (shape: 231, 1)
        output_path: Path to save visualization (optional)
        show: Whether to display plot
    """
    try:
        impedance = np.load(impedance_file)
        impedance = impedance.flatten()
        
        # Load frequency and target impedance from configs
        freq_file = Path(__file__).parent.parent / "configs" / "Frequency_data_hz.npy"
        target_file = Path(__file__).parent.parent / "configs" / "target_impedance.npy"
        
        if not freq_file.exists() or not target_file.exists():
            print(f"Warning: Config files not found. Plotting impedance only.")
            frequency = np.arange(len(impedance))
            target_impedance = None
        else:
            frequency = np.load(freq_file)
            target_impedance = np.load(target_file).flatten()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot generated impedance
        ax.loglog(frequency, impedance, marker='o', markersize=3, linestyle='-', 
                 linewidth=2, label='Generated Impedance', color='blue')
        
        # Plot target impedance if available
        if target_impedance is not None:
            ax.loglog(frequency, target_impedance, marker='s', markersize=2, linestyle='--', 
                     linewidth=2, label='Target Impedance', color='red')
        
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Impedance (Ohm)", fontsize=12)
        ax.set_title("Impedance Profile", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(1e-3, 1e2)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Impedance visualization saved to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        print(f"Error visualizing impedance: {e}")
    return None
