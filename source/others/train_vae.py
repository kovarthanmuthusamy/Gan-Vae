"""
Training script for Multi-Input Variational Autoencoder

Demonstrates how to use the MultiInputVAE model with sample data
"""

import sys
import torch
import torch.optim as optim
from typing import Tuple, Dict, Optional
from pathlib import Path

# Ensure the source directory is importable when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

# Import the model and loss function
from model.vae_multi_input import MultiInputVAE
from loss.vae_loss import VAELoss
from others.vae_logger import VAETrainingLogger
from others.dataloader import create_data_loaders, VAEDataset
from others.model_to_config import model_to_config_generic

# Import gamma monitoring for tracking attention usage
try:
    import sys
    from pathlib import Path
    scripts_path = Path(__file__).parent.parent.parent / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    from monitor_attention_gammas import log_gamma_values
    GAMMA_MONITORING_AVAILABLE = True
except ImportError:
    GAMMA_MONITORING_AVAILABLE = False
    print("Warning: Gamma monitoring not available. Install scripts/monitor_attention_gammas.py")


def compute_beta_annealing(current_epoch: int,
                          start_epoch: int,
                          end_epoch: int,
                          max_beta: float) -> float:
    """
    Compute β (beta) value for KL annealing schedule.
    
    Implements linear ramp from β=0 to β=max_beta:
    - Before start_epoch: β = 0 (pure autoencoder)
    - Between start_epoch and end_epoch: β increases linearly
    - After end_epoch: β = max_beta (full regularization)
    
    Args:
        current_epoch: Current training epoch (0-indexed)
        start_epoch: Epoch to start ramping β from 0
        end_epoch: Epoch to reach maximum β
        max_beta: Maximum β value (e.g., 0.1-0.5)
    
    Returns:
        Current β value for this epoch
    """
    if current_epoch < start_epoch:
        return 0.0
    elif current_epoch >= end_epoch:
        return max_beta
    else:
        # Linear interpolation between start and end
        progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
        return max_beta * progress


def ensure_channel_first(tensor: torch.Tensor, expected_channels: int) -> torch.Tensor:
    """Ensure tensors with spatial dims are channel-first."""
    if tensor.ndim != 4:
        return tensor
    if tensor.shape[1] == expected_channels:
        return tensor
    channel_dim = next((dim for dim in range(1, tensor.ndim) if tensor.shape[dim] == expected_channels), None)
    if channel_dim is None:
        return tensor
    permute_order = [0, channel_dim] + [dim for dim in range(1, tensor.ndim) if dim != channel_dim]
    return tensor.permute(*permute_order).contiguous()


# ============================================================
# HYPERPARAMETER CONFIGURATION
# Modify these values to tune the model training
# ============================================================

# --- Model Architecture ---
MODEL_LATENT_DIM = 128          # Dimensionality of latent space
MODEL_HIDDEN_DIM = 512          # Hidden dimension in encoder/decoder

# --- Training Configuration ---
TRAIN_NUM_EPOCHS = 100           # Number of training epochs
TRAIN_BATCH_SIZE = 64           # Batch size for training
TRAIN_LEARNING_RATE = 1e-5      # Learning rate for optimizer (reduced to prevent NaN)
TRAIN_SEED = 42                 # Random seed for reproducibility

# --- Loss Weights (for weighted loss calculation) ---
# ============================================================
# LOSS FUNCTION CONFIGURATION
# ============================================================
# Weights are carefully balanced based on:
#   1. Data characteristics (size, scale, normalization)
#   2. Loss function scales (SSIM ~0-1, BCE ~0-1, Cosine ~0-2, MSE ~0-44)
#   3. Task importance (occupancy for safety, impedance for control)
#
# Data sizes:
#   - Heatmap: 64×64×2 = 8,192 pixels, range [0, 3.35] (percentile normalized)
#   - Occupancy: 7×8×1 = 56 pixels, range [0, 1] (binary mask)
#   - Impedance: 231×1 = 231 values, range [-4.32, 2.36] (log normalized)
#
# Effective loss contributions (approximate):
#   - Heatmap: 2.0 × SSIM ≈ 2.0 × 0.3 = 0.6
#   - Occupancy: 5.0 × (0.5×BCE + 0.5×Dice) ≈ 2.5
#   - Impedance: 1.0 × (0.7×Cosine + 0.3×MSE) ≈ 1.0
#   - KL: 0.1 × (annealing 0→1.0) = 0→0.1 initially, growing during training
#
# Total reconstruction loss ≈ 4.0, KL contribution ≈ 0-0.1 (ramping up)
# ============================================================

# --- Loss Weights (carefully balanced for different scales and importance) ---
LOSS_HEATMAP_WEIGHT = 2.0       # Heatmap: SSIM produces smaller values, spatial context important
LOSS_OCCUPANCY_WEIGHT = 5.0     # Occupancy: Most critical (safety), smallest data needs emphasis
LOSS_IMPEDANCE_WEIGHT = 1.0     # Impedance: Base weight (internal composition adjusted below)
LOSS_KL_WEIGHT = 0.1            # KL divergence: Regularization strength (with annealing)

# --- Hybrid Loss Component Weights ---
OCCUPANCY_BCE_WEIGHT = 0.5      # Occupancy: BCE for pixel-level accuracy
OCCUPANCY_DICE_WEIGHT = 0.5     # Occupancy: Dice for region-level & class imbalance
IMPEDANCE_COSINE_WEIGHT = 0.7   # Impedance: Cosine for pattern/trend matching (more important)
IMPEDANCE_MSE_WEIGHT = 0.3      # Impedance: MSE for magnitude (less critical, log-normalized data)

# --- Beta Schedule (KL Annealing) ---
# Allows encoder to organize latent space before KL regularization kicks in
BETA_ANNEALING_ENABLED = True   # Enable/disable KL annealing
BETA_ANNEAL_START_EPOCH = 0     # Epoch to start annealing (0-indexed)
BETA_ANNEAL_END_EPOCH = 15      # Epoch to finish annealing (reaches max beta)
BETA_MAX = 1.0                  # Maximum beta value (changed from 0.5 to 1.0)

# --- Data Configuration ---
DATA_DIR = "source/data_norm"      # Path to dataset directory
DATA_NORMALIZE = False          # Whether to normalize data
DATA_TRAIN_SPLIT = 0.8          # Train/validation split ratio
DATA_NUM_WORKERS = 4            # Number of data loading workers

# --- Experiment Configuration ---
EXPERIMENT_DIR = "experiments/exp010"  # Experiment root directory
LOG_DIR = "logs"                  # Log directory relative to EXPERIMENT_DIR (or absolute path)
CHECKPOINT_DIR = "checkpoints"       # Checkpoint directory relative to EXPERIMENT_DIR (or absolute path)
CHECKPOINT_SAVE_INTERVAL = 10    # Save checkpoint every N epochs

# ============================================================


class VAETrainer:
    """Trainer class for MultiInputVAE"""
    
    def __init__(self, 
                 model: MultiInputVAE,
                 loss_fn: VAELoss,
                 lr: float = 1e-5,
                 device: Optional[torch.device] = None,
                 logger: Optional[VAETrainingLogger] = None):
        """
        Initialize the trainer
        
        Args:
            model: MultiInputVAE model instance
            loss_fn: VAELoss instance
            lr: Learning rate
            device: torch device (cuda or cpu)
            logger: Optional VAE logger for tracking metrics
        """
        self.model = model
        self.loss_fn = loss_fn
        self.logger = logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)
    
    def train_step(self,
                   heatmap: torch.Tensor,
                   occupancy: torch.Tensor,
                   impedance: torch.Tensor,
                   beta: float = 1.0) -> Dict[str, float]:
        """
        Single training step with optional KL annealing
        
        Args:
            heatmap: (batch_size, 2, 64, 64)
            occupancy: (batch_size, 1, 7, 8)
            impedance: (batch_size, 231)
            beta: KL weight multiplier for annealing (0.0 to 1.0)
        
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Check inputs for NaN
        if torch.isnan(heatmap).any():
            raise ValueError("NaN detected in heatmap input")
        if torch.isnan(occupancy).any():
            raise ValueError("NaN detected in occupancy input")
        if torch.isnan(impedance).any():
            raise ValueError("NaN detected in impedance input")
        
        # Forward pass
        outputs = self.model(heatmap, occupancy, impedance)
        
        # Calculate loss with annealed KL weight
        loss_dict = self.loss_fn(outputs, heatmap, occupancy, impedance, kl_weight_multiplier=beta)
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Return losses as scalars
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in loss_dict.items()}
    
    def eval_step(self,
                  heatmap: torch.Tensor,
                  occupancy: torch.Tensor,
                  impedance: torch.Tensor,
                  beta: float = 1.0) -> Dict[str, float]:
        """
        Single evaluation step with optional KL annealing
        
        Args:
            heatmap: (batch_size, 2, 64, 64)
            occupancy: (batch_size, 1, 7, 8)
            impedance: (batch_size, 231)
            beta: KL weight multiplier for annealing (0.0 to 1.0)
        
        Returns:
            Dictionary with loss values
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(heatmap, occupancy, impedance)
            
            # Calculate loss with annealed KL weight
            loss_dict = self.loss_fn(outputs, heatmap, occupancy, impedance, kl_weight_multiplier=beta)
        
        # Return losses as scalars
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in loss_dict.items()}
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")


def train_vae(num_epochs: int = TRAIN_NUM_EPOCHS,
              batch_size: int = TRAIN_BATCH_SIZE,
              latent_dim: int = MODEL_LATENT_DIM,
              hidden_dim: int = MODEL_HIDDEN_DIM,
              learning_rate: float = TRAIN_LEARNING_RATE,
              log_dir: Optional[str] = LOG_DIR,
              checkpoint_dir: Optional[str] = CHECKPOINT_DIR,
              exp_dir: str = EXPERIMENT_DIR,
              data_dir: str = DATA_DIR,
              num_workers: int = DATA_NUM_WORKERS,
              normalize: bool = DATA_NORMALIZE,
              seed: int = TRAIN_SEED,
              heatmap_weight: float = LOSS_HEATMAP_WEIGHT,
              occupancy_weight: float = LOSS_OCCUPANCY_WEIGHT,
              impedance_weight: float = LOSS_IMPEDANCE_WEIGHT,
              kl_weight: float = LOSS_KL_WEIGHT,
              train_split: float = DATA_TRAIN_SPLIT,
              beta_annealing_enabled: bool = BETA_ANNEALING_ENABLED,
              beta_anneal_start_epoch: int = BETA_ANNEAL_START_EPOCH,
              beta_anneal_end_epoch: int = BETA_ANNEAL_END_EPOCH,
              beta_max: float = BETA_MAX,
              occupancy_bce_weight: float = OCCUPANCY_BCE_WEIGHT,
              occupancy_dice_weight: float = OCCUPANCY_DICE_WEIGHT,
              impedance_cosine_weight: float = IMPEDANCE_COSINE_WEIGHT,
              impedance_mse_weight: float = IMPEDANCE_MSE_WEIGHT):
    """
    Training loop for MultiInputVAE with KL annealing support
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        latent_dim: Latent dimension
        hidden_dim: Hidden dimension
        learning_rate: Learning rate for optimizer
        log_dir: Directory for logging (default: exp_dir/logs)
        checkpoint_dir: Directory for checkpoints (default: exp_dir/checkpoints)
        exp_dir: Experiment directory (default: experiments/exp009)
        data_dir: Path to dataset directory (default: src/data_norm)
        num_workers: Number of data loading workers (default: 4)
        normalize: Whether to normalize data (default: False)
        seed: Random seed for reproducibility (default: 42)
        heatmap_weight: Weight for heatmap reconstruction loss (default: 2.0)
        occupancy_weight: Weight for occupancy reconstruction loss (default: 5.0)
        impedance_weight: Weight for impedance reconstruction loss (default: 1.0)
        kl_weight: Weight for KL divergence regularization (default: 0.1)
        train_split: Train/validation split ratio (default: 0.8)
        beta_annealing_enabled: Enable KL annealing (default: True)
        beta_anneal_start_epoch: Epoch to start annealing (default: 0)
        beta_anneal_end_epoch: Epoch to finish annealing (default: 15)
        beta_max: Maximum beta value (default: 1.0)
        occupancy_bce_weight: Weight for BCE component in occupancy loss (default: 0.5)
        occupancy_dice_weight: Weight for Dice component in occupancy loss (default: 0.5)
        impedance_cosine_weight: Weight for Cosine component in impedance loss (default: 0.7)
        impedance_mse_weight: Weight for MSE component in impedance loss (default: 0.3)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set default paths if not provided
    exp_path = Path(exp_dir)
    
    # Convert relative log_dir to absolute path under exp_dir
    if log_dir is None:
        log_dir = str(exp_path / "logs")
    elif not Path(log_dir).is_absolute():
        log_dir = str(exp_path / log_dir)
    
    # Convert relative checkpoint_dir to absolute path under exp_dir
    if checkpoint_dir is None:
        checkpoint_dir = str(exp_path / "checkpoints")
    elif not Path(checkpoint_dir).is_absolute():
        checkpoint_dir = str(exp_path / checkpoint_dir)
    
    print(f"Experiment directory: {exp_dir}")
    print(f"Logging to: {log_dir}")
    print(f"Checkpoints to: {checkpoint_dir}")
    
    # Print beta annealing configuration
    if beta_annealing_enabled:
        print(f"\n[KL ANNEALING SCHEDULE]")
        print(f"  Enabled: True")
        print(f"  Start epoch: {beta_anneal_start_epoch}")
        print(f"  End epoch: {beta_anneal_end_epoch}")
        print(f"  Max β: {beta_max}")
        print(f"  First {beta_anneal_start_epoch} epochs: β = 0 (pure autoencoder)")
        print(f"  Epochs {beta_anneal_start_epoch}-{beta_anneal_end_epoch}: Linear ramp to β = {beta_max}")
        print(f"  After epoch {beta_anneal_end_epoch}: β = {beta_max} (full regularization)\n")
    else:
        print(f"  KL Annealing: Disabled (using fixed KL weight = {kl_weight})\n")
    
    # Initialize model and loss
    model = MultiInputVAE(latent_dim=latent_dim, hidden_dim=hidden_dim)
    loss_fn = VAELoss(heatmap_weight=heatmap_weight, 
                      occupancy_weight=occupancy_weight,
                      impedance_weight=impedance_weight,
                      kl_weight=kl_weight,
                      occupancy_bce_weight=occupancy_bce_weight,
                      occupancy_dice_weight=occupancy_dice_weight,
                      impedance_cosine_weight=impedance_cosine_weight,
                      impedance_mse_weight=impedance_mse_weight)
    
    # Log initial gamma values (should all be 0)
    if GAMMA_MONITORING_AVAILABLE:
        print("\n" + "="*80)
        print("INITIAL ATTENTION GAMMA VALUES (all should be 0.0)")
        print("="*80)
        log_gamma_values(model, epoch=0)
    
    # Initialize logger
    logger = VAETrainingLogger(log_dir, checkpoint_dir=checkpoint_dir)
    print(f"✓ Logger initialized - Checkpoints will be saved every {CHECKPOINT_SAVE_INTERVAL} epochs")
    print(f"  Checkpoint directory: {logger.checkpoint_dir}")
    
    # Initialize trainer
    trainer = VAETrainer(model, loss_fn, lr=learning_rate, device=device, logger=logger)
    
    # Save model configuration to YAML
    try:
        temp_checkpoint_path = str(Path(exp_dir) / ".temp_model.pt")
        torch.save(model.state_dict(), temp_checkpoint_path)
        
        # Create config from checkpoint
        config_path = str(Path(exp_dir) / "config.yaml")
        hp_dict = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'heatmap_weight': heatmap_weight,
            'occupancy_weight': occupancy_weight,
            'impedance_weight': impedance_weight,
            'kl_weight': kl_weight,
            'data_dir': data_dir,
            'normalize': normalize,
            'train_split': train_split,
            'seed': seed,
            'beta_annealing_enabled': beta_annealing_enabled,
            'beta_anneal_start_epoch': beta_anneal_start_epoch,
            'beta_anneal_end_epoch': beta_anneal_end_epoch,
            'beta_max': beta_max,
        }
        config = model_to_config_generic(temp_checkpoint_path, config_path, additional_config=hp_dict)
        print(f"Configuration saved to: {config_path}")
        
        # Clean up temp file
        Path(temp_checkpoint_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")
    
    # Create data loaders
    try:
        print(f"\nLoading data from: {data_dir}")
        train_loader, val_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            train_split=train_split,
            seed=seed
        )
    except ValueError as e:
        print(f"Error loading data: {e}")
        print("Falling back to dummy data...")
        use_dummy_data = True
        num_samples = 64
        heatmap_data = torch.randn(num_samples, 2, 64, 64)
        occupancy_data = torch.rand(num_samples, 1, 7, 8)
        impedance_data = torch.randn(num_samples, 231)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Compute beta for KL annealing
        if beta_annealing_enabled:
            beta = compute_beta_annealing(
                current_epoch=epoch,
                start_epoch=beta_anneal_start_epoch,
                end_epoch=beta_anneal_end_epoch,
                max_beta=beta_max
            )
        else:
            beta = 1.0  # Use full KL weight
        
        # Initialize epoch losses
        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'heatmap_loss': 0.0,
            'occupancy_loss': 0.0,
            'impedance_loss': 0.0,
            'kl_loss': 0.0
        }
        
        num_batches = 0
        
        # Use real dataloader if available
        if 'train_loader' in locals():
            for batch_idx, batch in enumerate(train_loader):
                heatmap = ensure_channel_first(batch['heatmap'], expected_channels=2).to(device)
                occupancy = ensure_channel_first(batch['occupancy'], expected_channels=1).to(device)
                impedance = batch['impedance'].to(device)
                
                # Debug: print first batch statistics
                if epoch == 0 and batch_idx == 0:
                    print(f"\n[FIRST BATCH STATS]")
                    print(f"Heatmap   - shape: {heatmap.shape}, min: {heatmap.min():.4f}, max: {heatmap.max():.4f}, mean: {heatmap.mean():.4f}, std: {heatmap.std():.4f}")
                    print(f"Occupancy - shape: {occupancy.shape}, min: {occupancy.min():.4f}, max: {occupancy.max():.4f}, mean: {occupancy.mean():.4f}, std: {occupancy.std():.4f}")
                    print(f"Impedance - shape: {impedance.shape}, min: {impedance.min():.4f}, max: {impedance.max():.4f}, mean: {impedance.mean():.4f}, std: {impedance.std():.4f}\n")
                
                # Training step with annealed beta
                batch_losses = trainer.train_step(heatmap, occupancy, impedance, beta=beta)
                
                for key in epoch_losses:
                    epoch_losses[key] += batch_losses[key]
                
                num_batches += 1
        else:
            # Fall back to dummy data
            num_batches = num_samples // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                heatmap = ensure_channel_first(heatmap_data[start_idx:end_idx], expected_channels=2).to(device)
                occupancy = ensure_channel_first(occupancy_data[start_idx:end_idx], expected_channels=1).to(device)
                impedance = impedance_data[start_idx:end_idx].to(device)
                
                # Training step with annealed beta
                batch_losses = trainer.train_step(heatmap, occupancy, impedance, beta=beta)
                
                for key in epoch_losses:
                    epoch_losses[key] += batch_losses[key]
        
        # Average losses across batches
        for key in epoch_losses:
            epoch_losses[key] /= num_batches if num_batches > 0 else 1
        
        # Validation phase
        val_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'heatmap_loss': 0.0,
            'occupancy_loss': 0.0,
            'impedance_loss': 0.0,
            'kl_loss': 0.0
        }
        
        num_val_batches = 0
        
        if 'val_loader' in locals():
            for batch in val_loader:
                heatmap = ensure_channel_first(batch['heatmap'], expected_channels=2).to(device)
                occupancy = ensure_channel_first(batch['occupancy'], expected_channels=1).to(device)
                impedance = batch['impedance'].to(device)
                
                # Validation step with annealed beta
                val_batch_losses = trainer.eval_step(heatmap, occupancy, impedance, beta=beta)
                
                for key in val_losses:
                    val_losses[key] += val_batch_losses[key]
                
                num_val_batches += 1
            
            # Average validation losses
            for key in val_losses:
                val_losses[key] /= num_val_batches if num_val_batches > 0 else 1
        
        # Log metrics
        if logger:
            logger.log(epoch + 1,
                      epoch_losses['total_loss'],
                      epoch_losses['recon_loss'],
                      epoch_losses['kl_loss'],
                      epoch_losses['heatmap_loss'],
                      epoch_losses['occupancy_loss'],
                      epoch_losses['impedance_loss'])
        
        # Save checkpoint every N epochs
        if logger and (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            print(f"\n[Saving checkpoint at epoch {epoch + 1}]")
            logger.save_checkpoint(epoch + 1, trainer.model, trainer.optimizer)
        elif (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            print(f"\n[WARNING] Checkpoint not saved at epoch {epoch + 1} - logger is None")
        
        # Print epoch progress with beta value
        beta_display = compute_beta_annealing(
            current_epoch=epoch,
            start_epoch=beta_anneal_start_epoch,
            end_epoch=beta_anneal_end_epoch,
            max_beta=beta_max
        ) if beta_annealing_enabled else 1.0
        
        if 'val_loader' in locals() and (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | β={beta_display:.4f} | Val Loss: {val_losses['total_loss']:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} | β={beta_display:.4f} | Train Loss: {epoch_losses['total_loss']:.4f}")
        
        # Monitor gamma values every 10 epochs to track attention usage
        if GAMMA_MONITORING_AVAILABLE and (epoch + 1) % 10 == 0:
            log_gamma_values(model, epoch=epoch + 1)
    
    # Generate plots if logger is available
    if logger:
        logger.plot()
        logger.plot_loss_components()
        logger.print_statistics()
    
    # Final gamma monitoring summary
    if GAMMA_MONITORING_AVAILABLE:
        print("\n" + "="*80)
        print("FINAL ATTENTION GAMMA VALUES (after training)")
        print("="*80)
        log_gamma_values(model, epoch=num_epochs)
        
        # Count active attention layers
        active_count = 0
        total_count = 0
        for name, param in model.named_parameters():
            if 'gamma' in name:
                total_count += 1
                if abs(param.item()) > 0.1:
                    active_count += 1
        
        print(f"\n✅ Training Complete!")
        print(f"   {active_count}/{total_count} attention layers are active (|γ| > 0.1)")
        if active_count == 0:
            print(f"   ⚠️  No attention layers activated - cross-modal info may not be necessary")
        elif active_count < total_count * 0.3:
            print(f"   ℹ️  Only {active_count} attention layers active - partial usage is OK")
        else:
            print(f"   🎯 Most attention layers are active - cross-modal learning is working!")
    
    return trainer


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add workspace root to path for imports
    workspace_root = Path(__file__).parent.parent.parent
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    
    # Example usage - uses hyperparameters defined at the top of the file
    trainer = train_vae()
