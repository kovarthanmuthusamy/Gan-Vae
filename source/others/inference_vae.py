"""
Inference script for Multi-Input VAE
Generates samples from random latent vectors using the trained decoder

Usage:
    1. Configure parameters at the top of this file (CHECKPOINT_PATH, NUM_SAMPLES, etc.)
    2. Run: python3 source/others/inference_vae.py
    
    Or use command-line arguments:
    python3 source/others/inference_vae.py 
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import argparse
import json
import matplotlib
try:
    matplotlib.use('TkAgg')  # Interactive backend for displaying plots
except ImportError:
    matplotlib.use('Agg')   # Fallback for headless servers
import matplotlib.pyplot as plt
import matplotlib as mpl

# Ensure the source directory is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT 
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from source.model.vae_multi_input_simple import MultiInputVAE
from Data_Creation.csv_to_occupancy import visualize_occupancy_vector



# ============================================================
# INFERENCE CONFIGURATION
# Modify these values to change inference behavior
# ============================================================

# --- Model Configuration ---
CHECKPOINT_PATH = "experiments/exp018/checkpoints/checkpoint_epoch_200.pt"  # Path to trained model checkpoint
MODEL_LATENT_DIM = 160          # Latent dimension (must match training)

# --- Generation Configuration ---
NUM_SAMPLES = 20               # Number of samples to generate
OUTPUT_DIR = "experiments/exp018/visuals"  # Directory to save outputs
SAVE_DATA = True                # Save raw .npy data files
SAVE_PLOTS = True               # Save visualization plots

stat_type = "percentile"  # Choose which stats to use for normalization (global or percentile)

# --- Device Configuration ---
USE_CUDA = True                 # Use CUDA if available

# ============================================================
class VAEInference:
    """Inference engine for generating samples from trained VAE decoder"""
    
    def __init__(self, checkpoint_path: str, 
                 latent_dim: int = MODEL_LATENT_DIM,
                 device: Optional[torch.device] = None):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            latent_dim: Latent dimension (must match training)
            device: torch device (cuda or cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.latent_dim = latent_dim
        
        # Load normalization statistics from the dataset directory
        norm_stats_path = PROJECT_ROOT / "datasets/data_norm/normalization_stats.json"
        print(f"Loading normalization stats from: {norm_stats_path}")
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        
        # Extract values for denormalization
        print(f"Using {stat_type} stats for normalization")
        self.max_value_min = norm_stats["Max_value"][f"{stat_type}_min"]
        self.max_value_max = norm_stats["Max_value"][f"{stat_type}_max"]
        self.imp_log_mean = norm_stats["Impedance"]["log_mean"]
        self.imp_log_std = norm_stats["Impedance"]["log_std"]
        
        # Load frequency data and target impedance for visualization
        self.frequency = np.load(PROJECT_ROOT / 'configs/Frequency_data_hz.npy').squeeze()
        self.target_impedance = np.load(PROJECT_ROOT / 'configs/target_impedance.npy').squeeze()
        
        print(f"Normalization ranges:")
        print(f"  Max value: [{self.max_value_min:.4f}, {self.max_value_max:.4f}]")
        print(f"  Impedance (log): mean={self.imp_log_mean:.4f}, std={self.imp_log_std:.4f}")
        
        # Load model
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        self.model = MultiInputVAE(latent_dim=latent_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load standardization parameters (for compatibility, but not used for denorm)
        self.max_impedance_mean = checkpoint.get('max_impedance_mean', 0.0)
        self.max_impedance_std = checkpoint.get('max_impedance_std', 1.0)
       # self.model.set_standardization_params(self.max_impedance_mean, self.max_impedance_std)
        
        # Load latent space statistics from checkpoint (validation set)
        self.mu_mean = checkpoint.get('mu_mean', 0.0)
        self.mu_std = checkpoint.get('mu_std', 1.0)
        self.mu_min = checkpoint.get('mu_min', -3.0)
        self.mu_max = checkpoint.get('mu_max', 3.0)
        self.latent_stats = checkpoint.get('latent_stats', None)
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch: {self.epoch}")
        print(f"Using device: {self.device}")
        print(f"\nLatent space statistics (from validation set):")
        print(f"  μ: mean={self.mu_mean:.4f}, std={self.mu_std:.4f}, range=[{self.mu_min:.4f}, {self.mu_max:.4f}]")
        if self.latent_stats:
            print(f"  Per-modality latent stats:")
            for name, s in self.latent_stats.items():
                print(f"    {name:12s}: mu_mean={s['mu_mean']:.4f}, mu_std={s['mu_std']:.4f}")
        else:
            print(f"  No per-modality latent stats in checkpoint (will sample from N(0,1))")
    
    def generate_save(self, num_samples=1, out_dir="temp_visuals", latent_scale=None):
        """
        Generate and save samples from random latent vectors
        
        Args:
            num_samples: Number of samples to generate
            out_dir: Output directory
            latent_scale: Scale for sampling latent vectors. If None, uses mu_std from training.
                         Use smaller values (e.g., 0.1-0.5) if training mu values are small.
        
        Returns:
            Tuple of (heatmap_norm, max_impedance_denorm, occupancy, impedance_denorm) tensors
        """
        # Use training mu_std as default scale if not specified
        if latent_scale is None:
            latent_scale = self.mu_std
        
        with torch.no_grad():
            # Use the model's inference method which correctly handles
            # the hybrid latent space (Gaussian + Gumbel-Softmax for occupancy)
            heatmap_norm, occupancy, impedance_norm, max_impedance_norm = self.model.inference(
                num_samples, self.device, latent_stats=self.latent_stats
            )
            
            # Occupancy decoder already outputs sigmoid values in [0,1]
            # Apply threshold to create binary occupancy map
            occupancy_binary = (occupancy >= 0.5).float()
            
            # Denormalize max_impedance from [0, 1] to original range
            max_impedance_denorm = max_impedance_norm * (self.max_value_max - self.max_value_min) + self.max_value_min
            
            # Denormalize impedance from z-score to log scale
            impedance_denorm = impedance_norm * self.imp_log_std + self.imp_log_mean
            
            # Denormalize heatmap ch0 using min-max formula: x_raw = x_norm * (x_max - x_min) + x_min
            # Heatmap ch0 was normalized per-sample as x_norm = x_raw / per_sample_max, so x_min = 0
            heatmap_ch0_min = 0.0
            max_imp = max_impedance_denorm.view(-1, 1, 1, 1)
            heatmap_physical = heatmap_norm * (max_imp - heatmap_ch0_min) + heatmap_ch0_min
            
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU for saving
        heatmap_norm_cpu = heatmap_norm.cpu().numpy()
        heatmap_physical_cpu = heatmap_physical.cpu().numpy()
        max_impedance_cpu = max_impedance_denorm.cpu().numpy()
        occupancy_cpu = occupancy_binary.cpu().numpy()  # Use binary occupancy
        impedance_cpu = impedance_denorm.cpu().numpy()
        
        print(f"\nGenerated shapes:")
        print(f"  Heatmap (normalized): {heatmap_norm_cpu.shape}")
        print(f"  Heatmap (physical/denormalized): {heatmap_physical_cpu.shape}")
        print(f"  Max Impedance (denormalized): {max_impedance_cpu.shape}")
        print(f"  Occupancy: {occupancy_cpu.shape}")
        print(f"  Impedance (denormalized, log scale): {impedance_cpu.shape}")
        print(f"\nDenormalized value ranges:")
        print(f"  Max Impedance: [{max_impedance_cpu.min():.4f}, {max_impedance_cpu.max():.4f}]")
        print(f"  Impedance (log): [{impedance_cpu.min():.4f}, {impedance_cpu.max():.4f}]")
        
        # Save raw data
        for i in range(num_samples):
            ## it will create a folder for each sample to save the corresponding heatmap, occupancy and impedance data
            data_dir = output_path / "data_sample_{}".format(i)  
            data_dir.mkdir(exist_ok=True)
            
            # Save both normalized and denormalized outputs
            np.save(data_dir / f"heatmap_norm.npy", heatmap_norm_cpu[i])
            np.save(data_dir / f"heatmap_physical.npy", heatmap_physical_cpu[i])  # Denormalized
            np.save(data_dir / f"max_impedance.npy", max_impedance_cpu[i])  # Denormalized
            np.save(data_dir / f"occupancy_map.npy", occupancy_cpu[i])
            np.save(data_dir / f"impedance_profile.npy", impedance_cpu[i])  # Denormalized (log scale)
            
            print(f"\nSaved sample {i} data to: {data_dir}")
            print(f"  Max impedance (denorm): {max_impedance_cpu[i].item():.4f}")
            print(f"  Impedance log range: [{impedance_cpu[i].min():.4f}, {impedance_cpu[i].max():.4f}]")
            
            # Generate visualization
            self.visualize_sample(
                heatmap_physical=heatmap_physical_cpu[i],
                impedance_log=impedance_cpu[i],
                occupancy=occupancy_cpu[i],
                max_value=max_impedance_cpu[i].item(),
                sample_idx=i,
                output_dir=data_dir
            )
    
    def visualize_sample(self, heatmap_physical, impedance_log, occupancy, max_value, sample_idx, output_dir):
        """
        Visualize generated sample with subplots for heatmap, impedance, and occupancy
        
        Args:
            heatmap_physical: Denormalized physical heatmap (2, 64, 64)
            impedance_log: Denormalized impedance in log scale (231,)
            occupancy: Occupancy vector (52,) or compatible shape
            max_value: Denormalized max impedance value (scalar)
            sample_idx: Sample index for title
            output_dir: Directory to save visualization
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 5))
        
        # Subplot 1: Heatmap (Channel 0 - impedance channel)
        ax1 = plt.subplot(1, 3, 1)
        heatmap_ch0 = heatmap_physical[0]  # Extract channel 0
        cmap_22 = mpl.colormaps['jet'].resampled(22)
        im1 = ax1.imshow(
            heatmap_ch0,
            cmap=cmap_22,
            interpolation='bicubic',
            aspect='auto',
            origin='lower'
        )
        ax1.set_title(f'Generated Heatmap (Sample {sample_idx})', fontsize=14)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        # Display max_value (denormalized) on the heatmap
        ax1.text(0.98, 0.02, f'Max Value (denorm): {max_value:.4f}', 
                 transform=ax1.transAxes, fontsize=11, 
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Subplot 2: Impedance Profile
        ax2 = plt.subplot(1, 3, 2)
        # Convert from log scale to Ohm scale
        impedance_ohm = np.exp(impedance_log)
        # Plot target impedance
        ax2.loglog(self.frequency, self.target_impedance, 
                  linestyle='--', linewidth=2.5, label='Target Impedance', color='red')
        # Plot generated impedance
        ax2.loglog(self.frequency, impedance_ohm,
                  linestyle='-', linewidth=2.5, label='Generated Impedance', color='blue')
        ax2.set_ylim(1e-3, 1e2)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Impedance (Ohm)', fontsize=12)
        ax2.set_title(f'Impedance Profile (Sample {sample_idx})', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, which='both', alpha=0.3)
        
        # Subplot 3: Active capacitors from occupancy vector
        ax3 = plt.subplot(1, 3, 3)
        active_labels = visualize_occupancy_vector(occupancy.flatten())
        label_str = ', '.join(active_labels) if active_labels else 'None'
        ax3.axis('off')
        ax3.set_title(f'Active Capacitors ({len(active_labels)}) - Sample {sample_idx}', fontsize=14)
        ax3.text(0.5, 0.5, label_str, ha='center', va='center',
                 transform=ax3.transAxes, fontsize=10, wrap=True,
                 bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
        
        # Save and display the combined plot
        plt.tight_layout()
        output_path = Path(output_dir) / f'visualization_sample_{sample_idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved: {output_path}")
        plt.show()  # Display the plot
        plt.close(fig)
  



def main():
    parser = argparse.ArgumentParser(description='VAE Inference - Generate samples from trained model')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                      help=f'Path to model checkpoint (default: {CHECKPOINT_PATH})')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                      help=f'Directory to save generated samples (default: {OUTPUT_DIR})')
    parser.add_argument('--num-samples', type=int, default=NUM_SAMPLES,
                      help=f'Number of samples to generate (default: {NUM_SAMPLES})')
    parser.add_argument('--latent-dim', type=int, default=MODEL_LATENT_DIM,
                      help=f'Latent dimension (default: {MODEL_LATENT_DIM})')
    parser.add_argument('--latent-scale', type=float, default=None,
                      help='Scale for sampling latent vectors (default: use mu_std from checkpoint)')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage even if CUDA is available')
    
    args = parser.parse_args()
    
    # Set device
    if args.cpu:
        device = torch.device('cpu')
    elif USE_CUDA and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Inference Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Latent scale: {args.latent_scale if args.latent_scale else 'auto (from checkpoint)'}")
    print(f"  Device: {device}")
    
    # Initialize inference engine
    inference = VAEInference(
        checkpoint_path=args.checkpoint,
        latent_dim=args.latent_dim,
        device=device
    )
    
    # Generate and save samples
    inference.generate_save(
        out_dir=args.output_dir,
        num_samples=args.num_samples,
        latent_scale=args.latent_scale
    )
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
