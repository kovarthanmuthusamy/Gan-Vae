"""Training script for Multi-Input VAE"""

import sys
import torch
import torch.optim as optim
from dataclasses import dataclass, replace, asdict
from typing import Dict, Optional, Tuple
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
    latent_dim: int = 128
    seed: int = 42

    heatmap_weight: float = 5.0
    occupancy_weight: float = 2.0
    impedance_weight: float = 5.0
    kl_weight: float = 0.01
    occupancy_bce_weight: float = 0.5
    occupancy_dice_weight: float = 0.5
    impedance_cosine_weight: float = 0.7
    impedance_mse_weight: float = 0.3

    beta_annealing_enabled: bool = True
    beta_anneal_start_epoch: int = 0
    beta_anneal_end_epoch: int = 50
    beta_max: float = 1.0

    data_dir: str = "datasets/source/data_norm"
    normalize: bool = False
    train_split: float = 0.9
    num_workers: int = 4

    experiment_dir: str = "experiments/exp011"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 10

    attn_lr_multiplier: float = 4.0
    attn_lr_max_multiplier: float = 6.0
    attn_boost_threshold: float = 0.02
    attn_boost_patience: int = 5
    attn_boost_factor: float = 1.5

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
                 attn_lr_multiplier: float = 2.0, attn_lr_max_multiplier: float = 4.0):
        self.model = model
        self.loss_fn = loss_fn
        self.logger = logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_group_idx: Optional[int] = None
        self.attn_lr_base = lr * attn_lr_multiplier
        self.attn_lr_max = lr * attn_lr_max_multiplier

        base_params = []
        attn_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'gamma' in name:
                attn_params.append(param)
            else:
                base_params.append(param)

        param_groups = [{'params': base_params, 'lr': lr}]
        if attn_params:
            self.attn_group_idx = len(param_groups)
            param_groups.append({'params': attn_params, 'lr': self.attn_lr_base})

        self.optimizer = optim.Adam(param_groups)
        self.model.to(self.device)
    
    def train_step(self, heatmap: torch.Tensor, occupancy: torch.Tensor, 
                   impedance: torch.Tensor, beta: float = 1.0) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(heatmap, occupancy, impedance)
        loss_dict = self.loss_fn(outputs, heatmap, occupancy, impedance, kl_weight_multiplier=beta)
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def eval_step(self, heatmap: torch.Tensor, occupancy: torch.Tensor,
                  impedance: torch.Tensor, beta: float = 1.0) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(heatmap, occupancy, impedance)
            loss_dict = self.loss_fn(outputs, heatmap, occupancy, impedance, kl_weight_multiplier=beta)
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def save_checkpoint(self, path: str):
        torch.save({'model_state_dict': self.model.state_dict(), 
                   'optimizer_state_dict': self.optimizer.state_dict()}, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

    model = MultiInputVAE(latent_dim=cfg.latent_dim)
    loss_fn = VAELoss(
        heatmap_weight=cfg.heatmap_weight,
        occupancy_weight=cfg.occupancy_weight,
        impedance_weight=cfg.impedance_weight,
        kl_weight=cfg.kl_weight,
        occupancy_bce_weight=cfg.occupancy_bce_weight,
        occupancy_dice_weight=cfg.occupancy_dice_weight,
        impedance_cosine_weight=cfg.impedance_cosine_weight,
        impedance_mse_weight=cfg.impedance_mse_weight,
    )

    if GAMMA_MONITORING_AVAILABLE:
        logger.info("\nInitial gamma values:")
        log_gamma_values(model, epoch=0)

    trainer = VAETrainer(
        model,
        loss_fn,
        lr=cfg.learning_rate,
        device=device,
        logger=logger,
        attn_lr_multiplier=cfg.attn_lr_multiplier,
        attn_lr_max_multiplier=cfg.attn_lr_max_multiplier,
    )

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

    for epoch in range(cfg.num_epochs):
        beta = (
            compute_beta_annealing(epoch, cfg.beta_anneal_start_epoch, cfg.beta_anneal_end_epoch, cfg.beta_max)
            if cfg.beta_annealing_enabled else 1.0
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

            batch_losses = trainer.train_step(heatmap, occupancy, impedance, beta=beta)
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
            logger.save_checkpoint(epoch + 1, trainer.model, trainer.optimizer)

        logger.log_epoch_progress(
            epoch + 1,
            cfg.num_epochs,
            beta,
            epoch_losses['total_loss'],
            val_losses['total_loss'],
        )

        if GAMMA_MONITORING_AVAILABLE and (epoch + 1) % 10 == 0:
            log_gamma_values(model, epoch=epoch + 1)

    logger.plot()
    logger.plot_loss_components()
    logger.print_statistics()

    if GAMMA_MONITORING_AVAILABLE:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL ATTENTION GAMMA VALUES")
        logger.info("=" * 80)
        log_gamma_values(model, epoch=cfg.num_epochs)

        active_count = sum(
            1 for name, param in model.named_parameters()
            if 'gamma' in name and abs(param.item()) > 0.1
        )
        total_count = sum(1 for name, _ in model.named_parameters() if 'gamma' in name)

        logger.log_gamma_summary(active_count, total_count)

    return trainer


if __name__ == "__main__":
    train_vae()
