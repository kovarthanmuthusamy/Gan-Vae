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

# Ensure the source directory is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from model.vae_multi_input import MultiInputVAE


# ============================================================
# INFERENCE CONFIGURATION
# Modify these values to change inference behavior
# ============================================================

# --- Model Configuration ---
CHECKPOINT_PATH = "experiments/exp012/checkpoints/checkpoint_epoch_90.pt"  # Path to trained model checkpoint
MODEL_LATENT_DIM = 32          # Latent dimension (must match training)

# --- Generation Configuration ---
NUM_SAMPLES = 5               # Number of samples to generate
OUTPUT_DIR = "experiments/exp012/visuals"  # Directory to save outputs
SAVE_DATA = True                # Save raw .npy data files
SAVE_PLOTS = True               # Save visualization plots

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
        
        # Load model
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        self.model = MultiInputVAE(latent_dim=latent_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load standardization parameters
        self.max_impedance_mean = checkpoint.get('max_impedance_mean', 0.0)
        self.max_impedance_std = checkpoint.get('max_impedance_std', 1.0)
        self.model.set_standardization_params(self.max_impedance_mean, self.max_impedance_std)
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch: {self.epoch}")
        print(f"Max impedance stats: mean={self.max_impedance_mean:.4f}, std={self.max_impedance_std:.4f}")
        print(f"Using device: {self.device}")
    
    def generate_save(self, num_samples=1,out_dir="temp_visuals") :
        """
        Generate and save samples from random latent vectors
        
        Args:
            num_samples: Number of samples to generate
            out_dir: Output directory
        
        Returns:
            Tuple of (heatmap_norm, max_impedance_std, occupancy, impedance) tensors
        """
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            
            # Generate using decoder
            heatmap_norm, max_impedance_std, occupancy_logits, impedance = self.model.decoder(z)
            
            # Apply sigmoid to occupancy
            occupancy = torch.sigmoid(occupancy_logits)
            
            # Unstandardize max_impedance from Z-score to raw values
            max_impedance = self.model.unstandardize_max_impedance(max_impedance_std)
            
            # Compute physical heatmap: heatmap_physical = heatmap_norm * max_impedance
            heatmap_physical = heatmap_norm * max_impedance.view(-1, 1, 1, 1)
            
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU for saving
        heatmap_norm_cpu = heatmap_norm.cpu().numpy()
        heatmap_physical_cpu = heatmap_physical.cpu().numpy()
        max_impedance_cpu = max_impedance.cpu().numpy()
        occupancy_cpu = occupancy.cpu().numpy()
        impedance_cpu = impedance.cpu().numpy()
        
        print(f"Generated shapes:")
        print(f"  Heatmap (normalized): {heatmap_norm_cpu.shape}")
        print(f"  Heatmap (physical): {heatmap_physical_cpu.shape}")
        print(f"  Max Impedance: {max_impedance_cpu.shape}")
        print(f"  Occupancy: {occupancy_cpu.shape}")
        print(f"  Impedance: {impedance_cpu.shape}")
        print(f"  Max Impedance range: [{max_impedance_cpu.min():.4f}, {max_impedance_cpu.max():.4f}]")
        
        # Save raw data
        for i in range(num_samples):
            ## it will create a folder for each sample to save the corresponding heatmap, occupancy and impedance data
            data_dir = output_path / "data_sample_{}".format(i)  
            data_dir.mkdir(exist_ok=True)
            
            # Save both normalized and physical heatmaps
            np.save(data_dir / f"heatmap_norm.npy", heatmap_norm_cpu[i])
            np.save(data_dir / f"heatmap_physical.npy", heatmap_physical_cpu[i])
            np.save(data_dir / f"max_impedance.npy", max_impedance_cpu[i])
            np.save(data_dir / f"occupancy_map.npy", occupancy_cpu[i])
            np.save(data_dir / f"impedance_profile.npy", impedance_cpu[i])
            
            print(f"\nSaved sample {i} data to: {data_dir}")
            print(f"  Max impedance value: {max_impedance_cpu[i].item():.4f}")
        

  



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
        num_samples=args.num_samples
    )
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
