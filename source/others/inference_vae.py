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
CHECKPOINT_PATH = "experiments/exp010/checkpoints/epoch_100.pt"  # Path to trained model checkpoint
MODEL_LATENT_DIM = 128          # Latent dimension (must match training)
MODEL_HIDDEN_DIM = 512          # Hidden dimension (must match training)

# --- Generation Configuration ---
NUM_SAMPLES = 3               # Number of samples to generate
OUTPUT_DIR = "experiments/exp010/visuals"  # Directory to save outputs
SAVE_DATA = True                # Save raw .npy data files
SAVE_PLOTS = True               # Save visualization plots

# --- Device Configuration ---
USE_CUDA = True                 # Use CUDA if available

# ============================================================
class VAEInference:
    """Inference engine for generating samples from trained VAE decoder"""
    
    def __init__(self, checkpoint_path: str, 
                 latent_dim: int = MODEL_LATENT_DIM,
                 hidden_dim: int = MODEL_HIDDEN_DIM,
                 device: Optional[torch.device] = None):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            latent_dim: Latent dimension (must match training)
            hidden_dim: Hidden dimension (must match training)
            device: torch device (cuda or cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Load model
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        self.model = MultiInputVAE(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch: {self.epoch}")
        print(f"Using device: {self.device}")
    
    def generate_save(self, num_samples=1,out_dir="temp_visuals") :
        """
        Generate and save samples from random latent vectors
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Tuple of (heatmap, occupancy, impedance) tensors
        """
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            
            # Generate using decoder
            heatmap, occupancy, impedance = self.model.decoder(z)
            
            # Apply sigmoid to occupancy if it's logits
            if occupancy.min() < 0 or occupancy.max() > 1:
                occupancy = torch.sigmoid(occupancy)
            
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU for saving
        heatmap_cpu = heatmap.cpu().numpy()
        occupancy_cpu = occupancy.cpu().numpy()
        impedance_cpu = impedance.cpu().numpy()
        
        print(f"Generated shapes:")
        print(f"  Heatmap: {heatmap_cpu.shape}")
        print(f"  Occupancy: {occupancy_cpu.shape}")
        print(f"  Impedance: {impedance_cpu.shape}")
        
        # Save raw data
        for i in range(num_samples):
            ## it will create a folder for each sample to save the corresponding heatmap, occupancy and impedance data
            data_dir = output_path / "data_sample_{}".format(i)  
            data_dir.mkdir(exist_ok=True)
            
            np.save(data_dir / f"heatmap.npy", heatmap_cpu[i])
            np.save(data_dir / f"occupancy_map.npy", occupancy_cpu[i])
            np.save(data_dir / f"impedance_profile.npy", impedance_cpu[i])
            
            print(f"\nSaved {num_samples} data files to: {data_dir}")
        

  



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
    parser.add_argument('--hidden-dim', type=int, default=MODEL_HIDDEN_DIM,
                      help=f'Hidden dimension (default: {MODEL_HIDDEN_DIM})')
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
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Device: {device}")
    
    # Initialize inference engine
    inference = VAEInference(
        checkpoint_path=args.checkpoint,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
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
