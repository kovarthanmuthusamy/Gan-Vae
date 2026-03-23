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

from experiments.exp020.codes_1.vae_multi_input_simple import MultiInputVAE
from Data_Creation.csv_to_occupancy import visualize_occupancy_vector



# ============================================================
# INFERENCE CONFIGURATION
# Modify these values to change inference behavior
# ============================================================

# --- Model Configuration ---
CHECKPOINT_PATH = "experiments/exp020/checkpoints/checkpoint_epoch_300.pt"  # Path to trained model checkpoint
MODEL_LATENT_DIM = 132          # Latent dimension (must match training)

# --- Generation Configuration ---
NUM_SAMPLES = 20               # Number of samples to generate
OUTPUT_DIR = "experiments/exp020/visuals_2"  # Directory to save outputs
SAVE_DATA = True                # Save raw .npy data files
SAVE_PLOTS = True               # Save visualization plots

#stat_type = "percentile"  # Choose which stats to use for normalization (global or percentile)

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
       # print(f"Using {stat_type} stats for normalization")
        self.imp_log_mean = norm_stats["Impedance"]["log_mean"]
        self.imp_log_std = norm_stats["Impedance"]["log_std"]
        self.hm_log_mean = norm_stats["Heatmap"]["log_mean"]
        self.hm_log_std = norm_stats["Heatmap"]["log_std"]
        self.background_value = norm_stats.get("background_value", -3.6228)
        
        # Load frequency data, target impedance, and spatial board mask for visualization
        self.frequency = np.load(PROJECT_ROOT / 'configs/Frequency_data_hz.npy').squeeze()
        self.target_impedance = np.load(PROJECT_ROOT / 'configs/target_impedance.npy').squeeze()
        self.binary_mask = np.load(PROJECT_ROOT / 'configs/binary_mask.npy')  # (64, 64) bool
        
        print(f"Normalization ranges:")
        print(f"  Impedance (log): mean={self.imp_log_mean:.4f}, std={self.imp_log_std:.4f}")
        print(f"  Heatmap (log1p): mean={self.hm_log_mean:.4f}, std={self.hm_log_std:.4f}")
        
        # Load model
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        self.model = MultiInputVAE(latent_dim=latent_dim)

        # Robust checkpoint loading: for pure generation we only require decoder + latent heads.
        # This allows inference even if encoder architecture evolved after training.
        checkpoint_state = checkpoint['model_state_dict']
        model_state = self.model.state_dict()

        compatible_state = {}
        skipped_keys = []
        for key, value in checkpoint_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                skipped_keys.append(key)

        load_result = self.model.load_state_dict(compatible_state, strict=False)

        # Generation path depends on latent heads + decoders; require all of them.
        required_prefixes = (
            'heatmap_fc.',
            'heatmap_deconv.',
            'occupancy_decoder.',
            'impedance_decoder.',
            'heatmap_mu_private.',
            'occupancy_logits_private.',
            'impedance_mu_private.',
            'heatmap_mu_shared.',
            'occupancy_mu_shared.',
            'impedance_mu_shared.',
        )
        missing_required = [
            k for k in load_result.missing_keys
            if k.startswith(required_prefixes)
        ]
        if missing_required:
            raise RuntimeError(
                "Checkpoint is incompatible for inference: missing generation-critical "
                f"weights: {missing_required[:12]}"
            )

        loaded_count = len(compatible_state)
        total_count = len(checkpoint_state)
        print(f"Loaded {loaded_count}/{total_count} checkpoint tensors into model")
        if skipped_keys:
            preview = ', '.join(skipped_keys[:8])
            suffix = ' ...' if len(skipped_keys) > 8 else ''
            print(
                f"Skipped {len(skipped_keys)} incompatible tensors "
                f"(typically encoder-only for inference): {preview}{suffix}"
            )

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
            heatmap_zscore, occupancy, impedance_norm = self.model.inference(
                num_samples, self.device, latent_stats=self.latent_stats
            )
            
            # Occupancy decoder already outputs sigmoid values in [0,1]
            # Apply threshold to create binary occupancy map
            occupancy_binary = (occupancy >= 0.5).float()
            
            # Denormalize impedance from z-score to log scale
            # Dual-channel model outputs (B, 2, 231): Ch0=raw z-score, Ch1=first derivative
            if impedance_norm.dim() == 3 and impedance_norm.shape[1] == 2:
                impedance_deriv_norm = impedance_norm[:, 1, :]  # (B, 231) — derivative channel
                z_raw  = impedance_norm[:, 0, :]                # (B, 231) — raw channel

                # --- Integration Correction (Physics Constraint) ---
                # d[i] = z[i+1] - z[i]  (constant spacing, so cumsum inverts it)
                # z_integ[0]   = z_raw[0]
                # z_integ[i]   = z_raw[0] + sum(d[0..i-1])   for i >= 1
                d = impedance_deriv_norm                        # (B, 231)
                z_integ = torch.cat(
                    [z_raw[:, :1],
                     z_raw[:, :1] + torch.cumsum(d, dim=-1)[:, :-1]],
                    dim=-1
                )                                               # (B, 231)
                # Blend: cancel random noise while reinforcing structural peaks
                z_blended = 0.5 * (z_raw + z_integ)            # (B, 231)

                impedance_denorm       = z_blended * self.imp_log_std + self.imp_log_mean
                impedance_raw_denorm   = z_raw     * self.imp_log_std + self.imp_log_mean
                impedance_integ_denorm = z_integ   * self.imp_log_std + self.imp_log_mean
            else:
                impedance_deriv_norm   = None
                impedance_raw_denorm   = None
                impedance_integ_denorm = None
                impedance_denorm = impedance_norm * self.imp_log_std + self.imp_log_mean
            
            # Post-process heatmap: stamp sentinel on background using occupancy
            # Occupancy tells us which regions are foreground
            # For now, use the raw decoder output for foreground, sentinel for background
            # (The model was trained with masked loss — it never learned to predict background)
            
            # Denormalize heatmap from z-score to raw: z-score → log(1+x) → x
            heatmap_log = heatmap_zscore * self.hm_log_std + self.hm_log_mean
            heatmap_physical = torch.exp(heatmap_log) - 1.0
            heatmap_physical = torch.clamp(heatmap_physical, min=0.0)
            
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU for saving
        heatmap_zscore_cpu = heatmap_zscore.cpu().numpy()
        heatmap_physical_cpu = heatmap_physical.cpu().numpy()
        occupancy_cpu = occupancy_binary.cpu().numpy()  # Use binary occupancy
        impedance_cpu        = impedance_denorm.cpu().numpy()        # blended (main output)
        impedance_deriv_cpu  = impedance_deriv_norm.cpu().numpy()   if impedance_deriv_norm   is not None else None
        impedance_raw_cpu    = impedance_raw_denorm.cpu().numpy()   if impedance_raw_denorm   is not None else None
        impedance_integ_cpu  = impedance_integ_denorm.cpu().numpy() if impedance_integ_denorm is not None else None
        
        print(f"\nGenerated shapes:")
        print(f"  Heatmap (z-score): {heatmap_zscore_cpu.shape}")
        print(f"  Heatmap (physical/denormalized): {heatmap_physical_cpu.shape}")
        print(f"  Occupancy: {occupancy_cpu.shape}")
        print(f"  Impedance (denormalized, log scale): {impedance_cpu.shape}")
        print(f"\nDenormalized value ranges:")
        print(f"  Heatmap: [{heatmap_physical_cpu.min():.4f}, {heatmap_physical_cpu.max():.4f}]")
        print(f"  Impedance (log): [{impedance_cpu.min():.4f}, {impedance_cpu.max():.4f}]")
        
        # Save raw data
        for i in range(num_samples):
            ## it will create a folder for each sample to save the corresponding heatmap, occupancy and impedance data
            data_dir = output_path / "data_sample_{}".format(i)  
            data_dir.mkdir(exist_ok=True)
            
            # Save both normalized and denormalized outputs
            np.save(data_dir / f"heatmap_zscore.npy", heatmap_zscore_cpu[i])
            np.save(data_dir / f"heatmap_physical.npy", heatmap_physical_cpu[i])  # Denormalized
            np.save(data_dir / f"occupancy_map.npy", occupancy_cpu[i])
            np.save(data_dir / f"impedance_profile.npy", impedance_cpu[i])  # blended (log scale)
            if impedance_deriv_cpu is not None:
                np.save(data_dir / f"impedance_derivative.npy",  impedance_deriv_cpu[i])   # z-score derivative
                np.save(data_dir / f"impedance_raw.npy",          impedance_raw_cpu[i])     # type: ignore # ch0 raw (log scale)
                np.save(data_dir / f"impedance_integrated.npy",   impedance_integ_cpu[i])   # type: ignore # cumsum-reconstructed (log scale)
            
            print(f"\nSaved sample {i} data to: {data_dir}")
            print(f"  Heatmap physical range: [{heatmap_physical_cpu[i].min():.4f}, {heatmap_physical_cpu[i].max():.4f}]")
            print(f"  Impedance log range: [{impedance_cpu[i].min():.4f}, {impedance_cpu[i].max():.4f}]")
            
            # Generate visualization
            self.visualize_sample(
                heatmap_physical=heatmap_physical_cpu[i],
                impedance_log=impedance_cpu[i],
                occupancy=occupancy_cpu[i],
                sample_idx=i,
                output_dir=data_dir,
                impedance_derivative=impedance_deriv_cpu[i]   if impedance_deriv_cpu  is not None else None,
                impedance_raw_log=impedance_raw_cpu[i]        if impedance_raw_cpu     is not None else None,
                impedance_integ_log=impedance_integ_cpu[i]    if impedance_integ_cpu   is not None else None,
            )
    
    def visualize_sample(self, heatmap_physical, impedance_log, occupancy, sample_idx, output_dir,
                         impedance_derivative=None, impedance_raw_log=None, impedance_integ_log=None):
        """
        Visualize generated sample with subplots for heatmap, impedance, and occupancy.
        For dual-channel models the impedance plot shows:
          - raw ch0 reconstruction (thin dotted blue)
          - derivative-integrated reconstruction (thin dotted green)
          - blended physics-constraint result (solid blue, main output)
          - derivative overlaid on a twin y-axis (thin dotted orange)

        Args:
            heatmap_physical:    Denormalized physical heatmap (1, 64, 64)
            impedance_log:       Blended impedance in log scale (231,)
            occupancy:           Occupancy vector (52,) or compatible shape
            sample_idx:          Sample index for title
            output_dir:          Directory to save visualization
            impedance_derivative: Optional z-score first derivative (231,)
            impedance_raw_log:   Optional raw ch0 impedance in log scale (231,)
            impedance_integ_log: Optional cumsum-reconstructed impedance in log scale (231,)
        """
        #print(occupancy)
        # Create figure with subplots (3 cols always; derivative overlaid on impedance plot)
        n_cols = 3
        fig = plt.figure(figsize=(18, 5))
        
        # Subplot 1: Heatmap (Channel 0 - impedance channel)
        ax1 = plt.subplot(1, n_cols, 1)
        heatmap_ch0 = heatmap_physical[0]  # Extract channel 0 (1, 64, 64) -> (64, 64)
        # Apply spatial board mask: background pixels are masked and shown as white
        heatmap_masked = np.ma.masked_where(~self.binary_mask, heatmap_ch0)
        cmap_22 = mpl.colormaps['jet'].resampled(22).copy()
        cmap_22.set_bad(color='white')
        im1 = ax1.imshow(
            heatmap_masked,
            cmap=cmap_22,
            interpolation='bicubic',
            aspect='auto',
            origin='lower'
        )
        ax1.set_title(f'Generated Heatmap (Sample {sample_idx})', fontsize=14)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        cb = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        vmin, vmax = im1.get_clim()
        cb.set_ticks(np.linspace(vmin, vmax, 6))  # type: ignore # 6 ticks: always includes both endpoints
        
        # Subplot 2: Impedance Profile
        ax2 = plt.subplot(1, n_cols, 2)
        # Plot target impedance
        ax2.loglog(self.frequency, self.target_impedance,
                  linestyle='--', linewidth=2.5, label='Target Impedance', color='red')
        # Show only the integration-reconstructed curve (from derivative channel)
        if impedance_integ_log is not None:
            ax2.loglog(self.frequency, np.exp(impedance_integ_log),
                      linestyle='-', linewidth=2.5, color='mediumseagreen', label='Integrated (from derivative)')
        else:
            # Fallback: if no integ curve available, show the main output
            ax2.loglog(self.frequency, np.exp(impedance_log),
                      linestyle='-', linewidth=2.5, label='Generated Impedance', color='blue')
        ax2.set_ylim(1e-3, 1e2)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Impedance (Ohm)', fontsize=12)
        ax2.set_title(f'Impedance Profile (Sample {sample_idx})', fontsize=14)
        ax2.grid(True, which='both', alpha=0.3)

        # Overlay derivative on twin y-axis (thin dotted line)
        if impedance_derivative is not None:
            ax2b = ax2.twinx()
            ax2b.semilogx(self.frequency, impedance_derivative,
                          linestyle=':', linewidth=1.2, color='darkorange', label='Derivative (z-score)')
            ax2b.axhline(0, color='darkorange', linewidth=0.6, linestyle=':', alpha=0.4)
            ax2b.set_ylabel('First Derivative (z-score)', fontsize=11, color='darkorange')
            ax2b.tick_params(axis='y', labelcolor='darkorange')
            # Combined legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2b.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
        else:
            ax2.legend(fontsize=10)

        # Subplot 3: Active capacitors from occupancy vector
        ax3 = plt.subplot(1, n_cols, 3)
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
        plt.show(block=False)
        plt.pause(0.001)
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
