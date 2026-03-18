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
    num_epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1e-5
    latent_dim: int = 32
    seed: int = 42
    kl_weight: float = 0.01
    
    init_log_sigma_heatmap: float = -0.2
    init_log_sigma_occupancy: float = -1.2
    init_log_sigma_impedance: float = -0.2

    beta_annealing_enabled: bool = False
    beta_anneal_start_epoch: int = 0
    beta_anneal_end_epoch: int = 200
    beta_max: float = 0.2

    data_dir: str = "datasets/data_norm"
    normalize: bool = False
    train_split: float = 0.9
    num_workers: int = 4

    experiment_dir: str = "experiments/exp013"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 10
    resume_from_checkpoint: Optional[str] = "checkpoint_epoch_100.pt"  # Path to checkpoint file to resume from

    attn_lr_multiplier: float = 10.0
    attn_lr_max_multiplier: float = 15.0


    decoder_dropout_enabled: bool = True
    latent_dropout_prob: float = 0.15
    master_grid_dropout_prob: float = 0.10
    feature_dropout_prob: float = 0.05
    
    occupancy_loss_type: str = 'dice_bce'
    static_occupancy_weight: Optional[float] = 10.0
    static_weight_epochs: int = 100
    heatmap_gradient_weight: float = 0.1  # Weight for spatial gradient loss on heatmap
    impedance_gradient_weight: float = 0.1  # Weight for gradient loss to capture impedance peaks

    def resolve_paths(self) -> tuple[Path, Path, Path]:
        exp_path = Path(self.experiment_dir)
        log_path = Path(self.log_dir)
        if not log_path.is_absolute():
            log_path = exp_path / log_path
        checkpoint_path = Path(self.checkpoint_dir)
        if not checkpoint_path.is_absolute():
            checkpoint_path = exp_path / checkpoint_path
        return exp_path, log_path, checkpoint_path


LOSS_KEYS = ("total_loss", "recon_loss", "heatmap_loss", "max_impedance_loss", "occupancy_loss", "impedance_loss", "kl_loss")


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
                 attn_lr_multiplier: float = 4.0, attn_lr_max_multiplier: float = 6.0,
                 max_impedance_mean: float = 0.0, max_impedance_std: float = 1.0):
        self.model = model
        self.loss_fn = loss_fn
        self.logger = logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_lr_base = lr * attn_lr_multiplier
        self.attn_lr_max = lr * attn_lr_max_multiplier
        self.max_impedance_mean = max_impedance_mean
        self.max_impedance_std = max_impedance_std

        # Collect all trainable parameters (model + loss function uncertainty params)
        params = []
        for param in self.model.parameters():
            if param.requires_grad:
                params.append(param)
        
        for name, param in self.loss_fn.named_parameters():
            if param.requires_grad:
                params.append(param)
                if logger:
                    logger.info(f"Added loss parameter to optimizer: {name}")

        self.optimizer = optim.Adam(params, lr=lr)
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        
        # Set standardization parameters in model
        self.model.set_standardization_params(max_impedance_mean, max_impedance_std)
    
    def train_step(self, heatmap: torch.Tensor, max_impedance_std: torch.Tensor,
                   occupancy: torch.Tensor, impedance: torch.Tensor, beta: float = 1.0,
                   decoder_dropout_config: Optional[Dict[str, float]] = None,
                   compute_grad_norms: bool = False) -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Apply decoder weakening during training
        if decoder_dropout_config is not None:
            outputs = self.model.forward_with_decoder_dropout(
                heatmap, max_impedance_std, occupancy, impedance, **decoder_dropout_config)
        else:
            outputs = self.model(heatmap, max_impedance_std, occupancy, impedance)
            
        loss_dict = self.loss_fn(
            outputs, heatmap, max_impedance_std, occupancy, impedance,
            max_impedance_mean=self.max_impedance_mean,
            max_impedance_std=self.max_impedance_std,
            kl_weight_multiplier=beta,
            use_physical_loss=True
        )
        loss_dict['total_loss'].backward()
        
        # Compute gradient norms per output head if requested
        grad_norms = {}
        if compute_grad_norms:
            with torch.no_grad():
                # Heatmap decoder gradients
                heatmap_grads = [p.grad.norm().item() for p in self.model.decoder.heatmap_dec.parameters() 
                                if p.grad is not None]
                grad_norms['heatmap_grad_norm'] = sum(heatmap_grads) / max(len(heatmap_grads), 1)
                
                # Max impedance decoder gradients
                max_imp_grads = [p.grad.norm().item() for p in self.model.decoder.max_impedance_dec.parameters() 
                                if p.grad is not None]
                grad_norms['max_imp_grad_norm'] = sum(max_imp_grads) / max(len(max_imp_grads), 1)
                
                # Occupancy decoder gradients
                occ_grads = [p.grad.norm().item() for p in self.model.decoder.occupancy_dec.parameters() 
                            if p.grad is not None]
                grad_norms['occupancy_grad_norm'] = sum(occ_grads) / max(len(occ_grads), 1)
                
                # Impedance decoder gradients
                imp_grads = [p.grad.norm().item() for p in self.model.decoder.impedance_dec.parameters() 
                            if p.grad is not None]
                grad_norms['impedance_grad_norm'] = sum(imp_grads) / max(len(imp_grads), 1)
        
        # Gradient clipping for both model and loss function parameters
        all_params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        self.optimizer.step()
        
        result = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        result.update(grad_norms)
        return result
    
    def eval_step(self, heatmap: torch.Tensor, max_impedance_std: torch.Tensor,
                  occupancy: torch.Tensor, impedance: torch.Tensor, beta: float = 1.0) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(heatmap, max_impedance_std, occupancy, impedance)
            loss_dict = self.loss_fn(
                outputs, heatmap, max_impedance_std, occupancy, impedance,
                max_impedance_mean=self.max_impedance_mean,
                max_impedance_std=self.max_impedance_std,
                kl_weight_multiplier=beta,
                use_physical_loss=True
            )
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def save_checkpoint(self, path: str, epoch: Optional[int] = None):
        checkpoint: Dict[str, Any] = {
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),  # Save uncertainty parameters
            'max_impedance_mean': self.max_impedance_mean,  # Save standardization params
            'max_impedance_std': self.max_impedance_std      # Save standardization params
        }
        if epoch is not None:
            checkpoint['epoch'] = epoch
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return the epoch number"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            if "parameter group" in str(e) and self.logger:
                self.logger.info("Reinitializing optimizer (parameter groups changed)")
        
        if 'loss_fn_state_dict' in checkpoint:
            self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        
        return checkpoint.get('epoch', 0)

    def uncertainty_summary(self) -> Dict[str, float]:
        """Get current uncertainty parameter values"""
        with torch.no_grad():
            return {
                'sigma_heatmap': torch.exp(self.loss_fn.log_sigma_heatmap).item(),
                'sigma_occupancy': torch.exp(self.loss_fn.log_sigma_occupancy).item(),
                'sigma_impedance': torch.exp(self.loss_fn.log_sigma_impedance).item()
            }
    
    def kl_per_dimension_stats(self, heatmap: torch.Tensor, max_impedance_std: torch.Tensor,
                              occupancy: torch.Tensor, impedance: torch.Tensor) -> Dict[str, float]:
        """Compute KL divergence per latent dimension to detect collapsed dimensions"""
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encoder(heatmap, max_impedance_std, occupancy, impedance)
            # KL per dimension: -0.5 * (1 + log(σ²) - μ² - σ²)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim = kl_per_dim.mean(dim=0)  # Average over batch
            
            # Statistics
            kl_mean = kl_per_dim.mean().item()
            kl_std = kl_per_dim.std().item()
            kl_max = kl_per_dim.max().item()
            kl_min = kl_per_dim.min().item()
            
            # Count near-zero dims (collapsed posterior)
            near_zero = (kl_per_dim < 0.01).sum().item()
            active_dims = (kl_per_dim >= 0.01).sum().item()
            
            return {
                'kl_dim_mean': kl_mean,
                'kl_dim_std': kl_std,
                'kl_dim_max': kl_max,
                'kl_dim_min': kl_min,
                'kl_collapsed_dims': near_zero,
                'kl_active_dims': active_dims,
                'kl_active_ratio': active_dims / self.model.latent_dim
            }

    def log_attention_architecture(self):
        """Log attention layer placement"""
        if not hasattr(self, 'logger') or not self.logger:
            return
        
        self.logger.info("\n" + "="*80)
        self.logger.info("ATTENTION LAYER PLACEMENT")
        self.logger.info("="*80)
        
        attention_locations = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                attention_locations.append(name)
        
        if attention_locations:
            encoder_attn = [loc for loc in attention_locations if 'encoder' in loc]
            decoder_attn = [loc for loc in attention_locations if 'decoder' in loc]
            
            self.logger.info(f"Encoder layers: {len(encoder_attn)}")
            self.logger.info(f"Decoder layers: {len(decoder_attn)}")
            
            if len(encoder_attn) == 0:
                self.logger.info("WARNING: No encoder attention (all post-bottleneck)")
        else:
            self.logger.info("No attention layers found")
        
        self.logger.info("="*80)


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
    
    if cfg.decoder_dropout_enabled:
        logger.info(f"Decoder dropout enabled (latent: {cfg.latent_dropout_prob:.2f}, grid: {cfg.master_grid_dropout_prob:.2f})")
    
    total_recon_elements = 64 * 64 * 2 + 7 * 8 * 1 + 231
    kl_scaling_factor = total_recon_elements / cfg.latent_dim
    logger.info(f"KL scaling factor: {kl_scaling_factor:.1f} (base weight: {cfg.kl_weight})")
    
    max_impedance_mean, max_impedance_std_val = 0.0, 1.0
    
    model = MultiInputVAE(latent_dim=cfg.latent_dim)
    loss_fn = VAELoss(
        init_log_sigma_heatmap=cfg.init_log_sigma_heatmap,
        init_log_sigma_occupancy=cfg.init_log_sigma_occupancy,
        init_log_sigma_impedance=cfg.init_log_sigma_impedance,
        kl_weight=cfg.kl_weight,
        occupancy_loss_type=cfg.occupancy_loss_type,
        static_occupancy_weight=cfg.static_occupancy_weight,
        static_weight_epochs=cfg.static_weight_epochs,
        heatmap_gradient_weight=cfg.heatmap_gradient_weight,
        impedance_gradient_weight=cfg.impedance_gradient_weight
    )

    trainer = VAETrainer(
        model,
        loss_fn,
        lr=cfg.learning_rate,
        device=device,
        logger=logger,
        attn_lr_multiplier=cfg.attn_lr_multiplier,
        attn_lr_max_multiplier=cfg.attn_lr_max_multiplier,
        max_impedance_mean=max_impedance_mean,
        max_impedance_std=max_impedance_std_val,
    )

    start_epoch = 0
    if cfg.resume_from_checkpoint:
        checkpoint_path_to_load = Path(cfg.resume_from_checkpoint)
        if not checkpoint_path_to_load.is_absolute():
            checkpoint_path_to_load = checkpoint_path / cfg.resume_from_checkpoint
        
        if checkpoint_path_to_load.exists():
            logger.info(f"Resuming from: {checkpoint_path_to_load}")
            start_epoch = trainer.load_checkpoint(str(checkpoint_path_to_load))
            
            if start_epoch == 0:
                import re
                match = re.search(r'(?:checkpoint_)?epoch[_\s-]*(\d+)', checkpoint_path_to_load.name, re.IGNORECASE)
                if match:
                    start_epoch = int(match.group(1))
            
            logger.info(f"Resuming from epoch {start_epoch + 1}")

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

    for epoch in range(start_epoch, cfg.num_epochs):
        loss_fn.set_current_epoch(epoch)
        
        beta = (
            compute_beta_annealing(epoch, cfg.beta_anneal_start_epoch, cfg.beta_anneal_end_epoch, cfg.beta_max)
            if cfg.beta_annealing_enabled else cfg.beta_max
        )

        epoch_losses = init_loss_dict()

        for batch_idx, batch in enumerate(train_loader):
            heatmap = ensure_channel_first(batch['heatmap_norm'], expected_channels=2).to(device)
            max_impedance_std = batch['max_impedance_std'].to(device)
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

            # Compute gradient norms on first batch of epoch for monitoring
            compute_grad_norms = (batch_idx == 0 and (epoch + 1) % 10 == 0)
            batch_losses = trainer.train_step(heatmap, max_impedance_std, occupancy, impedance, beta=beta, 
                                             decoder_dropout_config=decoder_dropout_config,
                                             compute_grad_norms=compute_grad_norms)
            
            if compute_grad_norms and 'heatmap_grad_norm' in batch_losses:
                logger.info(f"\nEpoch {epoch+1} Gradient norms:")
                logger.info(f"  Heatmap: {batch_losses['heatmap_grad_norm']:.4f}")
                logger.info(f"  Occupancy: {batch_losses['occupancy_grad_norm']:.4f}")
                logger.info(f"  Impedance: {batch_losses['impedance_grad_norm']:.4f}")

            
            for key in epoch_losses:
                if key in batch_losses:
                    epoch_losses[key] += batch_losses[key]

        average_losses(epoch_losses, train_batches)

        val_losses = init_loss_dict()
        for batch in val_loader:
            heatmap = ensure_channel_first(batch['heatmap_norm'], expected_channels=2).to(device)
            max_impedance_std = batch['max_impedance_std'].to(device)
            occupancy = ensure_channel_first(batch['occupancy'], expected_channels=1).to(device)
            impedance = batch['impedance'].to(device)

            val_batch_losses = trainer.eval_step(heatmap, max_impedance_std, occupancy, impedance, beta=beta)
            for key in val_losses:
                val_losses[key] += val_batch_losses[key]

        average_losses(val_losses, val_batches)

        logger.log(
            epoch + 1,
            epoch_losses['total_loss'],
            epoch_losses['recon_loss'],
            epoch_losses['kl_loss'],
            epoch_losses['heatmap_loss'],
            epoch_losses['max_impedance_loss'],
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
                heatmap_sample = ensure_channel_first(sample_batch['heatmap_norm'], expected_channels=2).to(device)
                max_impedance_std_sample = sample_batch['max_impedance_std'].to(device)
                occupancy_sample = ensure_channel_first(sample_batch['occupancy'], expected_channels=1).to(device)  
                impedance_sample = sample_batch['impedance'].to(device)
                
                # Latent space diagnostics
                mu, logvar = trainer.model.encoder(heatmap_sample, max_impedance_std_sample, occupancy_sample, impedance_sample)
                mu_mean = mu.mean().item()
                mu_std = mu.std().item()
                logvar_mean = logvar.mean().item()
                sigma_mean = torch.exp(0.5 * logvar).mean().item()
                
                # Uncertainty parameters diagnostics
                sigma_hm = torch.exp(trainer.loss_fn.log_sigma_heatmap).item()
                sigma_occ = torch.exp(trainer.loss_fn.log_sigma_occupancy).item() 
                sigma_imp = torch.exp(trainer.loss_fn.log_sigma_impedance).item()
                
                # KL per dimension diagnostics
                kl_stats = trainer.kl_per_dimension_stats(heatmap_sample, max_impedance_std_sample, 
                                                          occupancy_sample, impedance_sample)
                
                logger.info(f"Epoch {epoch+1} diagnostics:")
                logger.info(f"  Latent space - μ: {mu_mean:.4f}±{mu_std:.4f}, log(σ²): {logvar_mean:.4f}, σ: {sigma_mean:.4f}")
                logger.info(f"  Uncertainties - σ_hm: {sigma_hm:.4f}, σ_occ: {sigma_occ:.4f}, σ_imp: {sigma_imp:.4f}")
                logger.info(f"  KL per dimension - mean: {kl_stats['kl_dim_mean']:.4f}, std: {kl_stats['kl_dim_std']:.4f}")
                logger.info(f"  KL per dimension - min: {kl_stats['kl_dim_min']:.4f}, max: {kl_stats['kl_dim_max']:.4f}")
                logger.info(f"  Active dimensions: {kl_stats['kl_active_dims']}/{trainer.model.latent_dim} ({kl_stats['kl_active_ratio']:.1%})")
                logger.info(f"  Collapsed dimensions (KL < 0.01): {kl_stats['kl_collapsed_dims']}")
                
                # Warning if too many dimensions collapsed
                if kl_stats['kl_active_ratio'] < 0.5:
                    logger.info(f"  ⚠️  WARNING: Over 50% of latent dimensions collapsed - consider stronger KL weight")
                
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



    logger.plot()
    logger.plot_loss_components()
    logger.print_statistics()



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
