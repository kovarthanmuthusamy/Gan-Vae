#!/usr/bin/env python3
"""
Simple script to plot training convergence from logs or CSV.
Usage: python plot_convergence.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import csv

# ============================================
# Configuration
# ============================================
CONFIG = {
    "experiment_name": "exp004",  # Change this to match your experiment
    "log_file": None,  # Optional: path to log file with printed output
    "csv_file": None,  # Optional: path to metrics.csv
}

def read_from_csv(csv_path):
    """Read metrics from CSV file"""
    epochs, loss_g, loss_d = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            loss_g.append(float(row['loss_g']))
            loss_d.append(float(row['loss_d']))
    return epochs, loss_g, loss_d

def read_from_log(log_path):
    """Read metrics from log file with printed output"""
    epochs, loss_g, loss_d = [], [], []
    pattern = r'Epoch (\d+)/\d+ \| Loss_G: ([-\d.]+) \| Loss_D: ([-\d.]+)'
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                loss_g.append(float(match.group(2)))
                loss_d.append(float(match.group(3)))
    return epochs, loss_g, loss_d

def plot_convergence(epochs, loss_g, loss_d, output_path):
    """Plot and save convergence graph"""
    epochs = np.array(epochs)
    loss_g = np.array(loss_g)
    loss_d = np.array(loss_d)
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss_g, 'b-', label='Generator Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, loss_d, 'r-', label='Discriminator Loss', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Convergence', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Plot saved to {output_path}")

if __name__ == "__main__":
    exp_name = CONFIG["experiment_name"]
    exp_dir = Path("experiments") / exp_name
    logs_dir = exp_dir / "logs"
    
    # Try to find data source
    csv_path = CONFIG["csv_file"] or (logs_dir / "metrics.csv")
    log_path = CONFIG["log_file"]
    
    epochs, loss_g, loss_d = None, None, None
    
    # Try CSV first
    if csv_path and Path(csv_path).exists():
        print(f"Reading from CSV: {csv_path}")
        epochs, loss_g, loss_d = read_from_csv(csv_path)
    
    # Try log file if CSV didn't work
    elif log_path and Path(log_path).exists():
        print(f"Reading from log: {log_path}")
        epochs, loss_g, loss_d = read_from_log(log_path)
    
    else:
        print("Error: No data source found!")
        print(f"  Tried CSV: {csv_path}")
        if log_path:
            print(f"  Tried log: {log_path}")
        print("\nOptions:")
        print("  1. Set CONFIG['csv_file'] to your metrics.csv path")
        print("  2. Set CONFIG['log_file'] to your training log file")
        print("  3. Or redirect training output: python train_main.py > training.log")
        exit(1)
    
    if not epochs:
        print("Error: No data found in the file!")
        exit(1)
    
    print(f"Found {len(epochs)} epochs of data")
    print(f"  Loss_G range: [{min(loss_g):.4f}, {max(loss_g):.4f}]")
    print(f"  Loss_D range: [{min(loss_d):.4f}, {max(loss_d):.4f}]")
    
    # Create output directory
    output_dir = exp_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "convergence.png"
    
    # Plot
    plot_convergence(epochs, loss_g, loss_d, output_path)
