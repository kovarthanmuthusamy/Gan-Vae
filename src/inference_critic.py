"""
Simple inference script for the Critic/Discriminator.
Loads a checkpoint from exp006 and evaluates real or generated samples.
Includes visualization of intermediate feature maps before fusion.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add repo-level paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.model_v1 import Critic, Generator

# ============================================
# Configuration
# ============================================
CONFIG = {
    "experiment_name": "exp006",
    "epoch": 100,  # Checkpoint epoch to load
    "latent_dim": 200,
    "shared_dim": 512,
    "device": "cuda:0",
}


def load_checkpoint(checkpoint_path, device):
    """Load generator and critic from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    G = Generator(CONFIG["latent_dim"], CONFIG["shared_dim"]).to(device)
    C = Critic().to(device)
    
    G.load_state_dict(checkpoint["G"])
    C.load_state_dict(checkpoint["D"])  # Discriminator/Critic saved as "D"
    
    G.eval()
    C.eval()
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    return G, C


def critic_inference_on_generated(critic, generator, num_samples=5, device: str | torch.device = "cuda:0"):
    """
    Run critic inference on generated samples.
    
    Args:
        critic: Loaded Critic model
        generator: Loaded Generator model
        num_samples: Number of samples to generate and evaluate
        device: Device to use
    
    Returns:
        scores: Critic scores for each sample
    """
    device = torch.device(device) if isinstance(device, str) else device
    
    with torch.no_grad():
        z = torch.randn(num_samples, CONFIG["latent_dim"], device=device)
        fake_heatmap, fake_occupancy, fake_impedance = generator(z)
        
        # Get critic scores
        scores = critic(fake_heatmap, fake_occupancy, fake_impedance)
    
    return scores.cpu().numpy().flatten()


def critic_inference_with_intermediates(critic, generator, device: str | torch.device = "cuda:0"):
    """
    Run critic inference on a single generated sample and return intermediate features.
    
    Args:
        critic: Loaded Critic model
        generator: Loaded Generator model
        device: Device to use
    
    Returns:
        score: Critic score
        intermediates: Dict of intermediate feature maps
        inputs: Dict of generator outputs (heatmap, occupancy, impedance)
    """
    device = torch.device(device) if isinstance(device, str) else device
    
    with torch.no_grad():
        z = torch.randn(1, CONFIG["latent_dim"], device=device)
        fake_heatmap, fake_occupancy, fake_impedance = generator(z)
        
        # Get critic scores with intermediates
        score, intermediates = critic.forward_with_intermediates(
            fake_heatmap, fake_occupancy, fake_impedance
        )
    
    # Convert intermediates to numpy
    intermediates_np = {k: v.cpu().numpy() for k, v in intermediates.items()}
    
    inputs = {
        'heatmap': fake_heatmap.cpu().numpy(),
        'occupancy': fake_occupancy.cpu().numpy(),
        'impedance': fake_impedance.cpu().numpy(),
    }
    
    return score.cpu().numpy().flatten()[0], intermediates_np, inputs


def plot_intermediate_features(intermediates, output_dir, prefix="generated"):
    """
    Plot intermediate feature maps from the critic.
    
    Args:
        intermediates: Dict of intermediate feature maps (numpy arrays)
        output_dir: Directory to save plots
        prefix: Prefix for filenames (e.g., 'generated' or 'real')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot the three heads before fusion (at 64x64 resolution)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Critic Input Heads Before Fusion ({prefix})", fontsize=14)
    
    # Heatmap head - channel 0 (impedance heatmap)
    heatmap = intermediates['heatmap'][0]  # (2, 64, 64)
    im0 = axes[0].imshow(heatmap[0], cmap='jet', interpolation='bilinear')
    axes[0].set_title("Heatmap Ch0\n(Impedance)")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Heatmap head - channel 1 (mask)
    im1 = axes[1].imshow(heatmap[1], cmap='gray', interpolation='bilinear')
    axes[1].set_title("Heatmap Ch1\n(Mask)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Occupancy head (upsampled)
    occ_up = intermediates['occ_up'][0, 0]  # (64, 64)
    im2 = axes[2].imshow(occ_up, cmap='Greys', interpolation='nearest')
    axes[2].set_title("Occupancy\n(Upsampled 64x64)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Impedance head (upsampled) - show mean across channels
    imp_up = intermediates['imp_up'][0]  # (4, 64, 64)
    im3 = axes[3].imshow(imp_up.mean(axis=0), cmap='viridis', interpolation='bilinear')
    axes[3].set_title("Impedance Proj\n(Mean of 4 ch, 64x64)")
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_heads_before_fusion.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / f'{prefix}_heads_before_fusion.png'}")
    
    # 2. Plot impedance projection detail (before and after upsample)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(f"Impedance Head Detail ({prefix})", fontsize=14)
    
    imp_proj = intermediates['imp_proj'][0]  # (4, 8, 8)
    for i in range(4):
        im = axes[i].imshow(imp_proj[i], cmap='viridis', interpolation='nearest')
        axes[i].set_title(f"Imp Proj Ch{i}\n(8x8)")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # Show upsampled mean
    im = axes[4].imshow(imp_up.mean(axis=0), cmap='viridis', interpolation='bilinear')
    axes[4].set_title("Upsampled Mean\n(64x64)")
    axes[4].axis('off')
    plt.colorbar(im, ax=axes[4], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_impedance_head_detail.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / f'{prefix}_impedance_head_detail.png'}")
    
    # 3. Plot fused features and conv progression
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Fusion and Conv Layers ({prefix})", fontsize=14)
    
    # Fused input - show first few channels
    fused = intermediates['fused'][0]  # (7, 64, 64)
    for i in range(4):
        ch_idx = min(i, fused.shape[0]-1)
        im = axes[0, i].imshow(fused[ch_idx], cmap='viridis', interpolation='bilinear')
        ch_names = ['Heat0', 'Heat1', 'Occ', 'Imp0', 'Imp1', 'Imp2', 'Imp3']
        axes[0, i].set_title(f"Fused Ch{ch_idx}\n({ch_names[ch_idx]}, 64x64)")
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)
    
    # After conv1 - show 4 channels
    after_conv1 = intermediates['after_conv1'][0]  # (16, 32, 32)
    for i in range(4):
        ch_idx = i * 4  # Show channels 0, 4, 8, 12
        im = axes[1, i].imshow(after_conv1[ch_idx], cmap='viridis', interpolation='bilinear')
        axes[1, i].set_title(f"Conv1 Ch{ch_idx}\n(32x32)")
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_fusion_conv_layers.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / f'{prefix}_fusion_conv_layers.png'}")
    
    # 4. Plot attention and deeper conv layers
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Attention and Deep Conv Layers ({prefix})", fontsize=14)
    
    # After conv2 (before attention)
    after_conv2 = intermediates['after_conv2'][0]  # (32, 16, 16)
    for i in range(4):
        ch_idx = i * 8  # Show channels 0, 8, 16, 24
        im = axes[0, i].imshow(after_conv2[ch_idx], cmap='viridis', interpolation='bilinear')
        axes[0, i].set_title(f"Conv2 Ch{ch_idx}\n(16x16)")
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)
    
    # After attention
    after_attn = intermediates['after_attn'][0]  # (32, 16, 16)
    for i in range(4):
        ch_idx = i * 8
        im = axes[1, i].imshow(after_attn[ch_idx], cmap='viridis', interpolation='bilinear')
        axes[1, i].set_title(f"After Attn Ch{ch_idx}\n(16x16)")
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_attention_layers.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / f'{prefix}_attention_layers.png'}")
    
    # 5. Plot final conv3 features
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Final Conv3 Features ({prefix}) - 8x8", fontsize=14)
    
    after_conv3 = intermediates['after_conv3'][0]  # (64, 8, 8)
    for i in range(8):
        row, col = i // 4, i % 4
        ch_idx = i * 8  # Show channels 0, 8, 16, 24, 32, 40, 48, 56
        im = axes[row, col].imshow(after_conv3[ch_idx], cmap='viridis', interpolation='nearest')
        axes[row, col].set_title(f"Conv3 Ch{ch_idx}")
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_final_conv3.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / f'{prefix}_final_conv3.png'}")


def critic_inference_on_real(critic, heatmap, occupancy, impedance, device: str | torch.device = "cuda:0"):
    """
    Run critic inference on real data samples.
    
    Args:
        critic: Loaded Critic model
        heatmap: Real heatmap data (batch, 2, 64, 64) or single sample
        occupancy: Real occupancy data (batch, 1, 7, 8) or single sample
        impedance: Real impedance data (batch, 231) or single sample
        device: Device to use
    
    Returns:
        scores: Critic scores for each sample
    """
    device = torch.device(device) if isinstance(device, str) else device
    
    # Convert to tensors if numpy arrays
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap).float()
    if isinstance(occupancy, np.ndarray):
        occupancy = torch.from_numpy(occupancy).float()
    if isinstance(impedance, np.ndarray):
        impedance = torch.from_numpy(impedance).float()
    
    # Add batch dimension if needed
    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)
    if occupancy.dim() == 3:
        occupancy = occupancy.unsqueeze(0)
    if impedance.dim() == 1:
        impedance = impedance.unsqueeze(0)
    
    heatmap = heatmap.to(device)
    occupancy = occupancy.to(device)
    impedance = impedance.to(device)
    
    with torch.no_grad():
        scores = critic(heatmap, occupancy, impedance)
    
    return scores.cpu().numpy().flatten()


def critic_inference_on_real_with_intermediates(critic, heatmap, occupancy, impedance, device: str | torch.device = "cuda:0"):
    """
    Run critic inference on real data sample and return intermediate features.
    
    Args:
        critic: Loaded Critic model
        heatmap: Real heatmap data (2, 64, 64) or (batch, 2, 64, 64)
        occupancy: Real occupancy data (1, 7, 8) or (batch, 1, 7, 8)
        impedance: Real impedance data (231,) or (batch, 231)
        device: Device to use
    
    Returns:
        score: Critic score
        intermediates: Dict of intermediate feature maps
    """
    device = torch.device(device) if isinstance(device, str) else device
    
    # Convert to tensors if numpy arrays
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap).float()
    if isinstance(occupancy, np.ndarray):
        occupancy = torch.from_numpy(occupancy).float()
    if isinstance(impedance, np.ndarray):
        impedance = torch.from_numpy(impedance).float()
    
    # Add batch dimension if needed
    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)
    if occupancy.dim() == 3:
        occupancy = occupancy.unsqueeze(0)
    if impedance.dim() == 1:
        impedance = impedance.unsqueeze(0)
    
    heatmap = heatmap.to(device)
    occupancy = occupancy.to(device)
    impedance = impedance.to(device)
    
    with torch.no_grad():
        score, intermediates = critic.forward_with_intermediates(heatmap, occupancy, impedance)
    
    # Convert intermediates to numpy
    intermediates_np = {k: v.cpu().numpy() for k, v in intermediates.items()}
    
    return score.cpu().numpy().flatten()[0], intermediates_np


def load_sample_data(sample_idx=0):
    """
    Load a sample from the normalized data directory.
    
    Returns:
        heatmap, occupancy, impedance as numpy arrays
    """
    data_dir = REPO_ROOT / "src" / "data_norm"
    
    heatmap_dir = data_dir / "heatmap"
    occ_dir = data_dir / "Occ_map"
    imp_dir = data_dir / "Imp"
    
    # Get file lists
    heatmap_files = sorted(list(heatmap_dir.glob("*.npy")))
    occ_files = sorted(list(occ_dir.glob("*.npy")))
    imp_files = sorted(list(imp_dir.glob("*.npy")))
    
    if sample_idx >= len(heatmap_files):
        raise ValueError(f"Sample index {sample_idx} out of range. Only {len(heatmap_files)} samples available.")
    
    heatmap = np.load(heatmap_files[sample_idx])
    occupancy = np.load(occ_files[sample_idx])
    impedance = np.load(imp_files[sample_idx])
    
    # Ensure correct shapes
    # Heatmap: (2, 64, 64)
    if heatmap.ndim == 2:
        heatmap = np.expand_dims(heatmap, 0)
    
    # Occupancy: (1, 7, 8)
    if occupancy.ndim == 2:
        occupancy = np.expand_dims(occupancy, 0)
    
    # Impedance: (231,) - flatten if needed
    impedance = impedance.flatten()
    
    return heatmap, occupancy, impedance


def main():
    device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build checkpoint path
    checkpoint_path = REPO_ROOT / "experiments" / CONFIG["experiment_name"] / "checkpoints" / f"epoch_{CONFIG['epoch']}.pt"
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        ckpt_dir = checkpoint_path.parent
        if ckpt_dir.exists():
            for f in sorted(ckpt_dir.glob("*.pt")):
                print(f"  - {f.name}")
        return
    
    # Load models
    G, C = load_checkpoint(checkpoint_path, device)
    
    # Output directory for visualizations
    output_dir = REPO_ROOT / "experiments" / CONFIG["experiment_name"] / "visuals" / "critic_intermediates"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving visualizations to: {output_dir}")
    
    print("\n" + "="*50)
    print("Critic Inference on Generated Samples")
    print("="*50)
    
    # Evaluate generated samples
    num_samples = 5
    gen_scores = critic_inference_on_generated(C, G, num_samples=num_samples, device=device)
    
    print(f"\nGenerated samples critic scores:")
    for i, score in enumerate(gen_scores):
        print(f"  Sample {i+1}: {score:.4f}")
    print(f"  Mean: {gen_scores.mean():.4f}")
    print(f"  Std:  {gen_scores.std():.4f}")
    
    # Get intermediates for one generated sample and plot
    print("\n" + "="*50)
    print("Plotting Intermediate Features (Generated Sample)")
    print("="*50)
    gen_score, gen_intermediates, gen_inputs = critic_inference_with_intermediates(C, G, device=device)
    print(f"Generated sample score: {gen_score:.4f}")
    plot_intermediate_features(gen_intermediates, output_dir, prefix="generated")
    
    print("\n" + "="*50)
    print("Critic Inference on Real Samples")
    print("="*50)
    
    # Evaluate real samples
    try:
        real_scores = []
        for i in range(min(5, 100)):  # Up to 5 real samples
            heatmap, occupancy, impedance = load_sample_data(sample_idx=i)
            score = critic_inference_on_real(C, heatmap, occupancy, impedance, device=device)
            real_scores.append(score[0])
            print(f"  Real sample {i+1}: {score[0]:.4f}")
        
        real_scores = np.array(real_scores)
        print(f"\nReal samples critic scores:")
        print(f"  Mean: {real_scores.mean():.4f}")
        print(f"  Std:  {real_scores.std():.4f}")
        
        # Get intermediates for one real sample and plot
        print("\n" + "="*50)
        print("Plotting Intermediate Features (Real Sample)")
        print("="*50)
        heatmap, occupancy, impedance = load_sample_data(sample_idx=0)
        real_score, real_intermediates = critic_inference_on_real_with_intermediates(
            C, heatmap, occupancy, impedance, device=device
        )
        print(f"Real sample score: {real_score:.4f}")
        plot_intermediate_features(real_intermediates, output_dir, prefix="real")
        
        print("\n" + "="*50)
        print("Summary")
        print("="*50)
        print(f"Real samples mean score:      {real_scores.mean():.4f}")
        print(f"Generated samples mean score: {gen_scores.mean():.4f}")
        print(f"Score difference (Real - Fake): {real_scores.mean() - gen_scores.mean():.4f}")
        print("\nNote: In WGAN, higher scores indicate 'more real' samples.")
        
    except Exception as e:
        print(f"Could not load real samples: {e}")
        print("Skipping real sample evaluation.")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
