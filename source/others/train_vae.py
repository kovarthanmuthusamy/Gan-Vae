"""Training script for Multi-Input VAE"""

import sys
import torch
import torch.optim as optim
from dataclasses import dataclass, replace, asdict
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from model.vae_multi_input import MultiInputVAE
from loss.vae_loss import VAELoss
from others.vae_logger import VAETrainingLogger
from others.dataloader import create_data_loaders, VAEDataset
from others.model_to_config import model_to_config_generic

try:
    scripts_path = Path(__file__).parent.parent.parent / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    from monitor_attention_gammas import log_gamma_values
    GAMMA_MONITORING_AVAILABLE = True
except ImportError:
    GAMMA_MONITORING_AVAILABLE = False


def compute_beta_annealing(current_epoch: int, start_epoch: int, 
                          end_epoch: int, max_beta: float) -> float:
    if current_epoch < start_epoch:
        return 0.0
    elif current_epoch >= end_epoch:
        return max_beta
    else:
        progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
        return max_beta * progress


def ensure_channel_first(tensor: torch.Tensor, expected_channels: int) -> torch.Tensor:
    if tensor.ndim != 4 or tensor.shape[1] == expected_channels:
        return tensor
    channel_dim = next((dim for dim in range(1, tensor.ndim) if tensor.shape[dim] == expected_channels), None)
    if channel_dim is None:
        return tensor
    permute_order = [0, channel_dim] + [dim for dim in range(1, tensor.ndim) if dim != channel_dim]
    return tensor.permute(*permute_order).contiguous()


@dataclass(frozen=True)
class TrainingConfig:
    num_epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 1e-5
    latent_dim: int = 32
    seed: int = 42

    heatmap_weight: float = 5.0  # DEPRECATED: Now using uncertainty-based weighting
    occupancy_weight: float = 2.0  # DEPRECATED: Now using uncertainty-based weighting  
    impedance_weight: float = 5.0  # DEPRECATED: Now using uncertainty-based weighting
    kl_weight: float = 0.01  # Good balance: KL contributes ~0.2 to total loss vs ~1.0 recon
                             # Raw KL values look high (~100) but effective contribution is reasonable
    
    # Uncertainty-based weighting parameters (Kendall et al.)
    # These control the initial uncertainty values for automatic loss balancing
    init_log_sigma_heatmap: float = -0.5    # Initial log(σ) for heatmap uncertainty
    init_log_sigma_occupancy: float = -0.3   # Initial log(σ) for occupancy uncertainty (higher = easier task)
    init_log_sigma_impedance: float = -0.5   # Initial log(σ) for impedance uncertainty
    
    occupancy_bce_weight: float = 0.5
    occupancy_dice_weight: float = 0.5
    impedance_cosine_weight: float = 0.7
    impedance_mse_weight: float = 0.3

    beta_annealing_enabled: bool = False
    beta_anneal_start_epoch: int = 0
    beta_anneal_end_epoch: int = 75
    beta_max: float = 0.4

    data_dir: str = "datasets/source/data_norm"
    normalize: bool = False
    train_split: float = 0.9
    num_workers: int = 4

    experiment_dir: str = "experiments/exp012"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 10
    resume_from_checkpoint: Optional[str] = "checkpoint_epoch_60.pt"  # Path to checkpoint file to resume from

    attn_lr_multiplier: float = 4.0
    attn_lr_max_multiplier: float = 6.0
    attn_boost_threshold: float = 0.02
    attn_boost_patience: int = 5
    attn_boost_factor: float = 1.5

    # Decoder weakening parameters to prevent overpowerful decoder
    decoder_dropout_enabled: bool = True
    latent_dropout_prob: float = 0.15       # Randomly zero out latent dimensions
    master_grid_dropout_prob: float = 0.10  # Dropout in shared feature grid
    feature_dropout_prob: float = 0.05      # Dropout in decoder intermediate layers
    
    # 🎯 NEW: Enhanced occupancy loss configuration 
    occupancy_loss_type: str = 'triple'  # 'bce', 'weighted_bce', 'dice', 'focal', 'dice_focal', 'combo', 'triple'
    occupancy_pos_weight: float = 15.0  # Weight positive class 15x higher
    focal_alpha: float = 0.25  # Focal loss alpha
    focal_gamma: float = 2.0   # Focal loss gamma
    static_occupancy_weight: Optional[float] = 5.0  # Static weight for first N epochs
    static_weight_epochs: int = 50  # Number of epochs to use static weight

    def resolve_paths(self) -> tuple[Path, Path, Path]:
        exp_path = Path(self.experiment_dir)
        log_path = Path(self.log_dir)
        if not log_path.is_absolute():
            log_path = exp_path / log_path
        checkpoint_path = Path(self.checkpoint_dir)
        if not checkpoint_path.is_absolute():
            checkpoint_path = exp_path / checkpoint_path
        return exp_path, log_path, checkpoint_path


LOSS_KEYS = ("total_loss", "recon_loss", "heatmap_loss", "occupancy_loss", "impedance_loss", "kl_loss")


def init_loss_dict() -> Dict[str, float]:
    return {key: 0.0 for key in LOSS_KEYS}


def average_losses(loss_dict: Dict[str, float], denom: int) -> None:
    if denom == 0:
        return
    for key in loss_dict:
        loss_dict[key] /= denom


class VAETrainer:
    def __init__(self, model: MultiInputVAE, loss_fn: VAELoss, lr: float = 1e-5,
                 device: Optional[torch.device] = None, logger: Optional[VAETrainingLogger] = None,
                 attn_lr_multiplier: float = 4.0, attn_lr_max_multiplier: float = 6.0):
        self.model = model
        self.loss_fn = loss_fn
        self.logger = logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_group_idx: Optional[int] = None
        self.attn_lr_base = lr * attn_lr_multiplier
        self.attn_lr_max = lr * attn_lr_max_multiplier

        base_params = []
        attn_params = []
        # Include model parameters
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'gamma' in name:
                attn_params.append(param)
            else:
                base_params.append(param)
        
        # Include loss function uncertainty parameters
        for name, param in self.loss_fn.named_parameters():
            if param.requires_grad:
                base_params.append(param)
                if logger:
                    logger.info(f"Added loss parameter to optimizer: {name}")

        param_groups = [{'params': base_params, 'lr': lr}]
        if attn_params:
            self.attn_group_idx = len(param_groups)
            param_groups.append({'params': attn_params, 'lr': self.attn_lr_base})

        self.optimizer = optim.Adam(param_groups)
        self.model.to(self.device)
        self.loss_fn.to(self.device)
    
    def train_step(self, heatmap: torch.Tensor, occupancy: torch.Tensor, 
                   impedance: torch.Tensor, beta: float = 1.0, 
                   decoder_dropout_config: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Apply decoder weakening during training
        if decoder_dropout_config is not None:
            outputs = self.model.forward_with_decoder_dropout(heatmap, occupancy, impedance, **decoder_dropout_config)
        else:
            outputs = self.model(heatmap, occupancy, impedance)
            
        loss_dict = self.loss_fn(outputs, heatmap, occupancy, impedance, kl_weight_multiplier=beta)
        loss_dict['total_loss'].backward()
        # Gradient clipping for both model and loss function parameters
        all_params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        self.optimizer.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def eval_step(self, heatmap: torch.Tensor, occupancy: torch.Tensor,
                  impedance: torch.Tensor, beta: float = 1.0) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(heatmap, occupancy, impedance)
            loss_dict = self.loss_fn(outputs, heatmap, occupancy, impedance, kl_weight_multiplier=beta)
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def save_checkpoint(self, path: str, epoch: Optional[int] = None):
        checkpoint: Dict[str, Any] = {
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict()  # Save uncertainty parameters
        }
        if epoch is not None:
            checkpoint['epoch'] = epoch
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return the epoch number to resume from"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Try to load optimizer state, but handle parameter group mismatches gracefully
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if hasattr(self, 'logger') and self.logger:
                self.logger.info("Successfully loaded optimizer state")
        except ValueError as e:
            if "parameter group" in str(e):
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info("⚠️  Optimizer parameter groups don't match checkpoint - reinitializing optimizer state")
                    self.logger.info("This is normal when model architecture or parameter grouping has changed")
                # Don't load optimizer state - it will start fresh but model weights are loaded
            else:
                raise e
        
        # Load loss function parameters if available (backward compatibility)
        if 'loss_fn_state_dict' in checkpoint:
            self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
            if hasattr(self, 'logger') and self.logger:
                self.logger.info("Loaded uncertainty parameters from checkpoint")
        else:
            if hasattr(self, 'logger') and self.logger:
                self.logger.info("No uncertainty parameters in checkpoint - using defaults")
        
        # Extract epoch number from checkpoint if available
        epoch = checkpoint.get('epoch', 0)
        return epoch

    def uncertainty_summary(self) -> Dict[str, float]:
       # """Get current uncertainty parameter values for monitoring"""
        with torch.no_grad():
            return {
                'sigma_heatmap': torch.exp(self.loss_fn.log_sigma_heatmap).item(),
                'sigma_occupancy': torch.exp(self.loss_fn.log_sigma_occupancy).item(),
                'sigma_impedance': torch.exp(self.loss_fn.log_sigma_impedance).item(),
                'log_sigma_heatmap': self.loss_fn.log_sigma_heatmap.item(),
                'log_sigma_occupancy': self.loss_fn.log_sigma_occupancy.item(),
                'log_sigma_impedance': self.loss_fn.log_sigma_impedance.item(),
            }

    def attention_gamma_stats(self) -> Tuple[float, int]:
        gammas = [param.detach().view(-1) for name, param in self.model.named_parameters() if 'gamma' in name]
        if not gammas:
            return 0.0, 0
        values = torch.cat(gammas)
        return float(values.abs().mean().item()), values.numel()

    def boost_attention_lr(self, factor: float) -> Optional[float]:
        if self.attn_group_idx is None:
            return None
        group = self.optimizer.param_groups[self.attn_group_idx]
        new_lr = min(group['lr'] * factor, self.attn_lr_max)
        if new_lr > group['lr']:
            group['lr'] = new_lr
            return new_lr
        return None


def train_vae(config: Optional[TrainingConfig] = None, **overrides):
    cfg = config or TrainingConfig()
    if overrides:
        cfg = replace(cfg, **overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_path, log_path, checkpoint_path = cfg.resolve_paths()

    logger = VAETrainingLogger(str(log_path), checkpoint_dir=str(checkpoint_path))
    logger.log_environment(device, exp_path, log_path, checkpoint_path)

    if cfg.beta_annealing_enabled:
        logger.log_beta_schedule(cfg.beta_anneal_start_epoch, cfg.beta_anneal_end_epoch, cfg.beta_max)
    else:
        logger.info(f"KL Annealing disabled (KL weight = {cfg.kl_weight})")
        
    # Log decoder weakening configuration
    if cfg.decoder_dropout_enabled:
        logger.info(f"Decoder weakening enabled:")
        logger.info(f"  - Latent dropout: {cfg.latent_dropout_prob:.3f}")
        logger.info(f"  - Master grid dropout: {cfg.master_grid_dropout_prob:.3f}")
        logger.info(f"  - Feature dropout: {cfg.feature_dropout_prob:.3f}")
    else:
        logger.info("Decoder weakening disabled")
    
    # Log KL scaling information
    # Calculate scaling factor (same as in loss function)
    total_recon_elements = 64 * 64 * 2 + 7 * 8 * 1 + 231  # 8,479 elements  
    kl_scaling_factor = total_recon_elements / cfg.latent_dim
    effective_kl_weight = cfg.kl_weight * kl_scaling_factor
    logger.info(f"Loss configuration:")
    logger.info(f"  KL loss scaling:")
    logger.info(f"    - Total reconstruction elements: {total_recon_elements}")
    logger.info(f"    - Latent dimensions: {cfg.latent_dim}")
    logger.info(f"    - KL scaling factor: {kl_scaling_factor:.1f}")
    logger.info(f"    - Base KL weight: {cfg.kl_weight}")
    logger.info(f"    - Effective KL weight: {effective_kl_weight:.3f}")
    logger.info(f"  Uncertainty-based weighting (Kendall et al.):")
    logger.info(f"    - Initial σ heatmap: {torch.exp(torch.tensor(cfg.init_log_sigma_heatmap)):.3f}")
    logger.info(f"    - Initial σ occupancy: {torch.exp(torch.tensor(cfg.init_log_sigma_occupancy)):.3f}")
    logger.info(f"    - Initial σ impedance: {torch.exp(torch.tensor(cfg.init_log_sigma_impedance)):.3f}")
    
    model = MultiInputVAE(latent_dim=cfg.latent_dim)
    loss_fn = VAELoss(
        init_log_sigma_heatmap=cfg.init_log_sigma_heatmap,
        init_log_sigma_occupancy=cfg.init_log_sigma_occupancy,
        init_log_sigma_impedance=cfg.init_log_sigma_impedance,
        kl_weight=cfg.kl_weight,
        # 🎯 NEW: Enhanced occupancy loss parameters
        occupancy_loss_type=cfg.occupancy_loss_type,
        occupancy_pos_weight=cfg.occupancy_pos_weight,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        static_occupancy_weight=cfg.static_occupancy_weight,
        static_weight_epochs=cfg.static_weight_epochs,
        occupancy_bce_weight=cfg.occupancy_bce_weight,
        occupancy_dice_weight=cfg.occupancy_dice_weight,
        impedance_cosine_weight=cfg.impedance_cosine_weight,
        impedance_mse_weight=cfg.impedance_mse_weight,
    )

    trainer = VAETrainer(
        model,
        loss_fn,
        lr=cfg.learning_rate,
        device=device,
        logger=logger,
        attn_lr_multiplier=cfg.attn_lr_multiplier,
        attn_lr_max_multiplier=cfg.attn_lr_max_multiplier,
    )

    # Handle checkpoint resumption
    start_epoch = 0
    if cfg.resume_from_checkpoint:
        checkpoint_path_to_load = Path(cfg.resume_from_checkpoint)
        if not checkpoint_path_to_load.is_absolute():
            checkpoint_path_to_load = checkpoint_path / cfg.resume_from_checkpoint
        
        logger.info(f"\nAttempting to load checkpoint from: {checkpoint_path_to_load}")
        logger.info(f"Checkpoint exists: {checkpoint_path_to_load.exists()}")
        
        if checkpoint_path_to_load.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path_to_load}")
            start_epoch = trainer.load_checkpoint(str(checkpoint_path_to_load))
            logger.info(f"Epoch from checkpoint: {start_epoch}")
            
            # Fallback: extract epoch from filename if not in checkpoint
            if start_epoch == 0:
                import re
                # Handle both "epoch_30.pt" and "checkpoint_epoch_30.pt" formats
                match = re.search(r'(?:checkpoint_)?epoch[_\s-]*(\d+)', checkpoint_path_to_load.name, re.IGNORECASE)
                if match:
                    start_epoch = int(match.group(1))
                    logger.info(f"Extracted epoch {start_epoch} from checkpoint filename: {checkpoint_path_to_load.name}")
                else:
                    logger.info(f"Could not extract epoch from filename: {checkpoint_path_to_load.name}")
            
            logger.info(f"==> Resuming training from epoch {start_epoch + 1} (continuing from {start_epoch})")
        else:
            logger.info(f"WARNING: Checkpoint not found at {checkpoint_path_to_load}")
            logger.info("Starting training from scratch")
            logger.info("Starting training from scratch")
    
    if GAMMA_MONITORING_AVAILABLE and start_epoch == 0:
        logger.info("\nInitial gamma values:")
        log_gamma_values(trainer.model, epoch=0)

    try:
        temp_path = exp_path / ".temp_model.pt"
        torch.save(model.state_dict(), temp_path)
        config_path = exp_path / "config.yaml"
        model_to_config_generic(str(temp_path), str(config_path), additional_config=asdict(cfg))
        temp_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"Config save warning: {e}")

    logger.log_data_source(cfg.data_dir)
    train_loader, val_loader = create_data_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        normalize=cfg.normalize,
        train_split=cfg.train_split,
        seed=cfg.seed,
    )

    train_batches = len(train_loader)
    val_batches = len(val_loader)
    low_gamma_epochs = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        # 🎯 NEW: Update loss function's current epoch for static weight logic
        loss_fn.set_current_epoch(epoch)
        
        beta = (
            compute_beta_annealing(epoch, cfg.beta_anneal_start_epoch, cfg.beta_anneal_end_epoch, cfg.beta_max)
            if cfg.beta_annealing_enabled else cfg.beta_max
        )

        epoch_losses = init_loss_dict()

        for batch_idx, batch in enumerate(train_loader):
            heatmap = ensure_channel_first(batch['heatmap'], expected_channels=2).to(device)
            occupancy = ensure_channel_first(batch['occupancy'], expected_channels=1).to(device)
            impedance = batch['impedance'].to(device)

            if epoch == 0 and batch_idx == 0:
                try:
                    dataset_size = len(train_loader.dataset)  # type: ignore[attr-defined]
                except (TypeError, AttributeError):
                    dataset_size = "unknown"
                logger.log_dataset_overview(dataset_size, train_batches)

            # Prepare decoder weakening configuration
            decoder_dropout_config = None
            if cfg.decoder_dropout_enabled:
                decoder_dropout_config = {
                    'latent_dropout_prob': cfg.latent_dropout_prob,
                    'master_grid_dropout_prob': cfg.master_grid_dropout_prob,
                    'feature_dropout_prob': cfg.feature_dropout_prob
                }

            batch_losses = trainer.train_step(heatmap, occupancy, impedance, beta=beta, 
                                             decoder_dropout_config=decoder_dropout_config)
            for key in epoch_losses:
                epoch_losses[key] += batch_losses[key]

        average_losses(epoch_losses, train_batches)

        val_losses = init_loss_dict()
        for batch in val_loader:
            heatmap = ensure_channel_first(batch['heatmap'], expected_channels=2).to(device)
            occupancy = ensure_channel_first(batch['occupancy'], expected_channels=1).to(device)
            impedance = batch['impedance'].to(device)

            val_batch_losses = trainer.eval_step(heatmap, occupancy, impedance, beta=beta)
            for key in val_losses:
                val_losses[key] += val_batch_losses[key]

        average_losses(val_losses, val_batches)

        mean_gamma, gamma_count = trainer.attention_gamma_stats()
        if gamma_count:
            logger.info(f"Attention |γ| mean: {mean_gamma:.4f} across {gamma_count} params")
            if mean_gamma < cfg.attn_boost_threshold:
                low_gamma_epochs += 1
                if low_gamma_epochs >= cfg.attn_boost_patience:
                    boosted_lr = trainer.boost_attention_lr(cfg.attn_boost_factor)
                    if boosted_lr is not None:
                        logger.info(f"Boosted attention LR to {boosted_lr:.2e} to encourage activation")
                    low_gamma_epochs = 0
            else:
                low_gamma_epochs = 0

        logger.log(
            epoch + 1,
            epoch_losses['total_loss'],
            epoch_losses['recon_loss'],
            epoch_losses['kl_loss'],
            epoch_losses['heatmap_loss'],
            epoch_losses['occupancy_loss'],
            epoch_losses['impedance_loss'],
        )

        if (epoch + 1) % cfg.checkpoint_interval == 0:
            checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_file), epoch=epoch + 1)
            logger.info(f"Saved checkpoint: {checkpoint_file}")

        logger.log_epoch_progress(
            epoch + 1,
            cfg.num_epochs,
            beta,
            epoch_losses['total_loss'],
            val_losses['total_loss'],
        )
        
        # Log latent diagnostics and uncertainty parameters every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                # Sample a batch for diagnostics
                sample_batch = next(iter(train_loader))
                heatmap_sample = ensure_channel_first(sample_batch['heatmap'], expected_channels=2).to(device)
                occupancy_sample = ensure_channel_first(sample_batch['occupancy'], expected_channels=1).to(device)  
                impedance_sample = sample_batch['impedance'].to(device)
                
                # Latent space diagnostics
                mu, logvar = trainer.model.encode(heatmap_sample, occupancy_sample, impedance_sample)
                mu_mean = mu.mean().item()
                mu_std = mu.std().item()
                logvar_mean = logvar.mean().item()
                sigma_mean = torch.exp(0.5 * logvar).mean().item()
                
                # Uncertainty parameters diagnostics
                sigma_hm = torch.exp(trainer.loss_fn.log_sigma_heatmap).item()
                sigma_occ = torch.exp(trainer.loss_fn.log_sigma_occupancy).item() 
                sigma_imp = torch.exp(trainer.loss_fn.log_sigma_impedance).item()
                
                logger.info(f"Epoch {epoch+1} diagnostics:")
                logger.info(f"  Latent space - μ: {mu_mean:.4f}±{mu_std:.4f}, log(σ²): {logvar_mean:.4f}, σ: {sigma_mean:.4f}")
                logger.info(f"  Uncertainties - σ_hm: {sigma_hm:.4f}, σ_occ: {sigma_occ:.4f}, σ_imp: {sigma_imp:.4f}")
                
                # Uncertainty interpretation
                uncertainties = {'heatmap': sigma_hm, 'occupancy': sigma_occ, 'impedance': sigma_imp}
                easiest_task = min(uncertainties, key=lambda x: uncertainties[x])
                hardest_task = max(uncertainties, key=lambda x: uncertainties[x])
                logger.info(f"  Task difficulty - Easiest: {easiest_task} (σ={uncertainties[easiest_task]:.4f}), Hardest: {hardest_task} (σ={uncertainties[hardest_task]:.4f})")
                
                # Check for posterior collapse indicators
                if abs(mu_mean) < 0.01 and mu_std < 0.1 and sigma_mean < 0.1:
                    logger.info("⚠️  WARNING: Potential posterior collapse detected (latents too small)")
                elif abs(mu_mean) > 3.0 or mu_std > 3.0:
                    logger.info("⚠️  WARNING: Latent values very large - may need to reduce KL weight")

        if GAMMA_MONITORING_AVAILABLE and (epoch + 1) % 10 == 0:
            log_gamma_values(trainer.model, epoch=epoch + 1)

    logger.plot()
    logger.plot_loss_components()
    logger.print_statistics()

    if GAMMA_MONITORING_AVAILABLE:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL ATTENTION GAMMA VALUES")
        logger.info("=" * 80)
        log_gamma_values(trainer.model, epoch=cfg.num_epochs)

        active_count = sum(
            1 for name, param in trainer.model.named_parameters()
            if 'gamma' in name and abs(param.item()) > 0.1
        )
        total_count = sum(1 for name, _ in trainer.model.named_parameters() if 'gamma' in name)

        logger.log_gamma_summary(active_count, total_count)    
    # Log final uncertainty parameters
    logger.info("\n" + "="*80)
    logger.info("FINAL UNCERTAINTY PARAMETERS (KENDALL ET AL.)")
    logger.info("="*80)
    uncertainty_summary = trainer.uncertainty_summary()
    logger.info(f"Final learned uncertainties (σ):")
    logger.info(f"  Heatmap: {uncertainty_summary['sigma_heatmap']:.4f} (log_σ: {uncertainty_summary['log_sigma_heatmap']:.4f})")
    logger.info(f"  Occupancy: {uncertainty_summary['sigma_occupancy']:.4f} (log_σ: {uncertainty_summary['log_sigma_occupancy']:.4f})")
    logger.info(f"  Impedance: {uncertainty_summary['sigma_impedance']:.4f} (log_σ: {uncertainty_summary['log_sigma_impedance']:.4f})")
    
    # Interpret results
    uncertainties = {
        'heatmap': uncertainty_summary['sigma_heatmap'],
        'occupancy': uncertainty_summary['sigma_occupancy'], 
        'impedance': uncertainty_summary['sigma_impedance']
    }
    easiest_task = min(uncertainties, key=lambda x: uncertainties[x])
    hardest_task = max(uncertainties, key=lambda x: uncertainties[x])
    
    logger.info(f"\nTask difficulty analysis:")
    logger.info(f"  Easiest task: {easiest_task} (σ = {uncertainties[easiest_task]:.4f})")
    logger.info(f"  Hardest task: {hardest_task} (σ = {uncertainties[hardest_task]:.4f})")
    logger.info(f"  Interpretation: Lower σ = model is more confident = task is easier to learn")
    return trainer


if __name__ == "__main__":
    train_vae()
