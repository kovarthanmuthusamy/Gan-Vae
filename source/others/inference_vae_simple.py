"""
Simple inference script for Multi-Input VAE (vae_multi_input copy.py)
Generate samples from trained model
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

# Import simplified model
import importlib.util
model_path = SOURCE_DIR / "model" / "vae_multi_input copy.py"
spec = importlib.util.spec_from_file_location("vae_model", model_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {model_path}")
vae_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vae_model)
MultiInputVAE = vae_model.MultiInputVAE


# ============================================================
# CONFIGURATION
# ============================================================
CHECKPOINT_PATH = "experiments/exp015/checkpoints/checkpoint_300.pt"
NUM_SAMPLES = 5
OUTPUT_DIR = "experiments/exp015/generated_samples"
LATENT_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# INFERENCE
# ============================================================
def load_model(checkpoint_path, latent_dim, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = MultiInputVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model


def generate_samples(model, num_samples, device):
    """Generate samples from random latent vectors"""
    print(f"\nGenerating {num_samples} samples...")
    
    with torch.no_grad():
        # Use the model's inference method
        heatmap, occupancy, impedance, maxvalue = model.inference(
            num_samples=num_samples,
            device=device
        )
    
    # Move to CPU for saving/visualization
    heatmap_np = heatmap.cpu().numpy()
    occupancy_np = occupancy.cpu().numpy()
    impedance_np = impedance.cpu().numpy()
    maxvalue_np = maxvalue.cpu().numpy()
    
    print(f"✓ Generated samples:")
    print(f"  Heatmap: {heatmap_np.shape}")
    print(f"  Occupancy: {occupancy_np.shape}")
    print(f"  Impedance: {impedance_np.shape}")
    print(f"  Max Value: {maxvalue_np.shape}")
    
    return heatmap_np, occupancy_np, impedance_np, maxvalue_np


def save_samples(heatmap, occupancy, impedance, maxvalue, output_dir):
    """Save generated samples"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_samples = heatmap.shape[0]
    
    for i in range(num_samples):
        # Create sample directory
        sample_dir = output_path / f"sample_{i:03d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Save raw data
        np.save(sample_dir / "heatmap.npy", heatmap[i])
        np.save(sample_dir / "occupancy.npy", occupancy[i])
        np.save(sample_dir / "impedance.npy", impedance[i])
        np.save(sample_dir / "maxvalue.npy", maxvalue[i])
        
        # Create visualization
        visualize_sample(
            heatmap[i], occupancy[i], impedance[i], maxvalue[i],
            save_path=sample_dir / "visualization.png"
        )
        
        print(f"  ✓ Saved sample {i} to {sample_dir}")


def visualize_sample(heatmap, occupancy, impedance, maxvalue, save_path):
    """Create visualization of generated sample"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Heatmap (channel 0)
    ax = axes[0, 0]
    im = ax.imshow(heatmap[0], cmap='jet', aspect='auto')
    ax.set_title('Heatmap (Channel 0)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    
    # Occupancy
    ax = axes[0, 1]
    im = ax.imshow(occupancy[0], cmap='binary', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Occupancy Map')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    
    # Impedance profile
    ax = axes[1, 0]
    ax.plot(impedance, linewidth=1.5)
    ax.set_title('Impedance Profile (Normalized)')
    ax.set_xlabel('Frequency Point')
    ax.set_ylabel('Normalized Value')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Max value display
    ax = axes[1, 1]
    ax.text(0.5, 0.6, f'Max Value', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.4, f'{maxvalue.item():.4f}', ha='center', va='center', fontsize=24)
    ax.text(0.5, 0.25, '(Normalized)', ha='center', va='center', fontsize=12, style='italic')
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print("="*80)
    print("SIMPLE VAE INFERENCE")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print("="*80)
    
    # Load model
    model = load_model(CHECKPOINT_PATH, LATENT_DIM, DEVICE)
    
    # Generate samples
    heatmap, occupancy, impedance, maxvalue = generate_samples(
        model, NUM_SAMPLES, DEVICE
    )
    
    # Save samples
    print(f"\nSaving samples to: {OUTPUT_DIR}")
    save_samples(heatmap, occupancy, impedance, maxvalue, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"Generated {NUM_SAMPLES} samples")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
