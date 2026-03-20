"""
Simple training script for Multi-Input VAE (vae_multi_input_simple.py)
Streamlined version without complex features
"""

import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT 
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

# Import simplified model (vae_multi_input_simple.py)
from source.model.vae_multi_input_simple import MultiInputVAE
from source.others.dataloader import create_data_loaders
from source.others.vae_logger import VAETrainingLogger


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Model
    latent_dim: int = 160  # Total latent dim (private + shared)
    
    # Training
    num_epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-5
    
    # Loss weights
    heatmap_weight: float = 1.0
    occupancy_weight: float = 1.0
    impedance_weight: float = 0.5
    maxvalue_weight: float = 0.1
    
    # Gumbel-Softmax temperature annealing for occupancy
    gumbel_temp_start: float = 1.0    # Start warm (soft samples)
    gumbel_temp_end: float = 0.1      # Anneal to cold (hard samples)
    gumbel_anneal_epochs: int = 100   # Anneal over this many epochs
    
    
    # Free bits: minimum KL per latent dimension (prevents private-space collapse)
    free_bits: float = 0.5  # nats per dim; 0 = disabled, 0.25-1.0 typical
    
    # Beta-VAE Annealing (KL weight annealing)
    use_beta_annealing: bool = True  # Set to True to enable beta annealing
    beta_start_epoch: int = 0        # Start annealing from epoch 0
    beta_end_epoch: int = 100         # Ramp up over 100 epochs
    beta_initial: float = 0.0        # Start with 0 KL weight
    beta_final: float = 0.0005       # Ramp to final KL weight (same as beta_weight for fixed KL)
    
    # Data
    data_dir: str = "datasets/data_norm"
    train_split: float = 0.9
    num_workers: int = 4
    
    # Checkpoints

    experiment_dir: str = "experiments/exp018"  # Directory to save checkpoints and logs
    checkpoint_interval: int = 10
    resume_checkpoint: str = ""#"experiments/exp018/checkpoints/checkpoint_epoch_100.pt"  # Architecture changed (Gumbel-Softmax) — old checkpoints incompatible
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # max value denormalization — must match datasets/data_norm normalization_stats.json (p0.1/p99.9)
    max_value_min: float = 1.5898871421813965
    max_value_max: float = 10.52902603149414



# ============================================================
# CHECKPOINT LOADING
# ============================================================
def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """
    Load checkpoint and restore training state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint to
    
    Returns:
        start_epoch: Epoch to resume from
        best_val_loss: Best validation loss from checkpoint
    """
    print(f"\n{'='*80}")
    print(f"RESUMING FROM CHECKPOINT")
    print(f"{'='*80}")
    print(f"Loading: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model state loaded")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
    
    # Get training state
    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"✓ Resuming from epoch {start_epoch}")
    print(f"✓ Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*80}\n")
    
    return start_epoch, best_val_loss


# ============================================================
# BETA ANNEALING
# ============================================================
def compute_beta(epoch, config):
    """
    Compute annealed beta (KL weight) for current epoch
    
    Args:
        epoch: Current epoch (0-indexed)
        config: Configuration object
    
    Returns:
        beta: Annealed KL weight
    """
    if not config.use_beta_annealing:
        return config.beta_final  # Use final beta if annealing is disabled
    
    if epoch < config.beta_start_epoch:
        return config.beta_initial
    elif epoch >= config.beta_end_epoch:
        return config.beta_final
    else:
        # Linear annealing
        progress = (epoch - config.beta_start_epoch) / (config.beta_end_epoch - config.beta_start_epoch)
        return config.beta_initial + progress * (config.beta_final - config.beta_initial)

# ============================================================
# LOSS FUNCTION
# ============================================================
def vae_loss(recon_heatmap, recon_occupancy, recon_impedance, recon_maxvalue,
             target_heatmap, target_occupancy, target_impedance, target_maxvalue,
             mu_gaussian, logvar_gaussian, occ_logits, beta, config):
    """
    Hybrid VAE loss: Reconstruction + Gaussian KL + Gumbel KL
    
    Args:
        mu_gaussian, logvar_gaussian: Gaussian parts (heatmap, impedance, maxvalue privates + shared)
        occ_logits: (B, occ_priv, 2) Gumbel-Softmax logits for occupancy
        beta: Annealed KL weight
    """
    # combined los for max_value and heatmap: MSE on denormalized values (emphasizes peaks more)
    max_value_min = config.max_value_min
    max_value_max = config.max_value_max
    # Denorm max_value: percentile min-max → raw scale factor (per-sample)
    max_imp_denorm   = (target_maxvalue * (max_value_max - max_value_min) + max_value_min).view(-1, 1, 1)
    max_imp_denorm_p = (recon_maxvalue  * (max_value_max - max_value_min) + max_value_min).view(-1, 1, 1)

    # Channel 0: min-max denorm then MSE — x_raw = x_norm * (x_max - x_min) + x_min
    # Heatmap ch0 was normalized per-sample as x_norm = x_raw / per_sample_max, so x_min = 0, x_max = max_imp_denorm
    heatmap_ch0_min = 0.0
    loss_heatmap_ch0 = F.mse_loss(
        recon_heatmap[:, 0] * (max_imp_denorm_p - heatmap_ch0_min) + heatmap_ch0_min,
        target_heatmap[:, 0] * (max_imp_denorm   - heatmap_ch0_min) + heatmap_ch0_min,
        reduction='mean'
    )
    # Channel 1: constant binary mask — apply separately so it doesn't dilute ch0 gradients
    loss_heatmap_ch1 = F.mse_loss(recon_heatmap[:, 1], target_heatmap[:, 1], reduction='mean')
    loss_heatmap = loss_heatmap_ch0 + loss_heatmap_ch1

    #occupancy loss: binary cross-entropy with logits (numerically stable)
    loss_occupancy = F.binary_cross_entropy(
        recon_occupancy.clamp(1e-7, 1 - 1e-7),
        target_occupancy.clamp(0.0, 1.0),
        reduction='mean'
    )
    
    # Peak-sensitive impedance loss: weight peaks 10x more than flat regions
    # Compute per-point error (delta=0.4 scaled for z-score clip range ~[-2.6, +2.1])
    imp_error = F.huber_loss(recon_impedance, target_impedance, delta=0.4, reduction='none')  # (B, 231)
    # Detect peaks: how much each point deviates from sample mean (global, not local)
    with torch.no_grad():
        imp_mean = target_impedance.mean(dim=1, keepdim=True)  # (B, 1)
        deviation = (target_impedance - imp_mean).abs()  # (B, 231)
        # Normalize deviation to [0, 1] per sample, then scale: flat=1x, peaks=10x
        dev_max = deviation.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        peak_weight = 1.0 + 9.0 * (deviation / dev_max)  # (B, 231), range [1, 10]
    loss_impedance_recon = (imp_error * peak_weight).mean() * 5

    # Smoothness loss: penalize differences between adjacent frequency points
    # Forces generated curve slopes to match target slopes, suppressing noise/jaggedness
    recon_diff = recon_impedance[:, 1:] - recon_impedance[:, :-1]     # (B, 230)
    target_diff = target_impedance[:, 1:] - target_impedance[:, :-1]  # (B, 230)
    loss_impedance_smooth = F.mse_loss(recon_diff, target_diff)

    loss_impedance = loss_impedance_recon + 2.0 * loss_impedance_smooth
        
    # Total reconstruction loss (weighted)
    recon_loss = (config.heatmap_weight * loss_heatmap +
                  config.occupancy_weight * loss_occupancy +
                  config.impedance_weight * loss_impedance )
    
    # === Gaussian KL divergence with free bits (prevents private-space collapse) ===
    # Per-dim KL: 0.5 * (μ² + σ² - log(σ²) - 1)
    kl_per_dim = -0.5 * (1 + logvar_gaussian - mu_gaussian.pow(2) - logvar_gaussian.exp())  # (B, D)
    kl_per_dim = kl_per_dim.mean(dim=0)  # (D,) average over batch
    if config.free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=config.free_bits)  # floor at λ nats
    kl_gaussian = kl_per_dim.sum()
    
    # === Gumbel KL divergence (for occupancy) ===
    # KL(q(z|x) || Uniform(0.5, 0.5))
    # = Σ_i Σ_k q_ik * log(q_ik / p_k)  where p_k = 0.5
    # = Σ_i Σ_k q_ik * (log(q_ik) + log(2))
    occ_probs = F.softmax(occ_logits, dim=-1)  # (B, occ_priv, 2)
    log_probs = torch.log(occ_probs + 1e-8)
    # KL from uniform prior: each category has prior prob 0.5
    log_prior = torch.log(torch.tensor(0.5, device=occ_logits.device))
    kl_gumbel = (occ_probs * (log_probs - log_prior)).sum(dim=-1).sum(dim=-1)  # (B,)
    kl_gumbel = kl_gumbel.mean()  # Average over batch
    
    # Total KL
    kl_loss = kl_gaussian + kl_gumbel
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'kl_gaussian': kl_gaussian,
        'kl_gumbel': kl_gumbel,
        'heatmap_loss': loss_heatmap,
        'occupancy_loss': loss_occupancy,
        'impedance_loss': loss_impedance
    }


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, train_loader, optimizer, config, epoch, beta, temperature):
    """Train for one epoch with annealed beta and Gumbel temperature"""
    model.train()
    total_losses = {'total_loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0,
                   'kl_gaussian': 0.0, 'kl_gumbel': 0.0,
                   'heatmap_loss': 0.0, 'occupancy_loss': 0.0, 
                   'impedance_loss': 0.0,  'beta': beta}
    
    # Track mu and logvar statistics (Gaussian parts only)
    mu_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    logvar_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    std_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    
    # Track per-modality private stats
    modality_stats = {
        'heatmap':   {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0},
        'occupancy': {'prob1_mean': 0.0, 'prob1_std': 0.0},  # Gumbel: track P(1) 
        'impedance': {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0},
        'maxvalue':  {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0}
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} (β={beta:.5f}, τ={temperature:.3f})")
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        heatmap = batch['heatmap_norm'].to(config.device)
        occupancy = batch['occupancy'].to(config.device)
        impedance = batch['impedance'].to(config.device)
        maxvalue = batch['max_impedance_std'].to(config.device)
        
        # Ensure correct shapes
        if heatmap.dim() == 3:
            heatmap = heatmap.unsqueeze(1)
        if impedance.dim() == 1:
            impedance = impedance.unsqueeze(0)
        if maxvalue.dim() == 1:
            maxvalue = maxvalue.unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        recon_hm, recon_occ, recon_imp, recon_mv, mu_gaussian, logvar_gaussian, occ_logits, mod_stats = model(
            heatmap, occupancy, impedance, maxvalue, temperature=temperature
        )
        
        # Compute loss with annealed beta (hybrid: Gaussian KL + Gumbel KL)
        losses = vae_loss(recon_hm, recon_occ, recon_imp, recon_mv,
                         heatmap, occupancy, impedance, maxvalue,
                         mu_gaussian, logvar_gaussian, occ_logits, beta, config)
        
        
        # Track mu and logvar statistics (Gaussian parts only)
        with torch.no_grad():
            mu_stats['mean'] += mu_gaussian.mean().item()
            mu_stats['std'] += mu_gaussian.std().item()
            mu_stats['min'] = min(mu_stats['min'], mu_gaussian.min().item())
            mu_stats['max'] = max(mu_stats['max'], mu_gaussian.max().item())
            
            logvar_stats['mean'] += logvar_gaussian.mean().item()
            logvar_stats['std'] += logvar_gaussian.std().item()
            logvar_stats['min'] = min(logvar_stats['min'], logvar_gaussian.min().item())
            logvar_stats['max'] = max(logvar_stats['max'], logvar_gaussian.max().item())
            
            std = torch.exp(logvar_gaussian / 2)
            std_stats['mean'] += std.mean().item()
            std_stats['std'] += std.std().item()
            std_stats['min'] = min(std_stats['min'], std.min().item())
            std_stats['max'] = max(std_stats['max'], std.max().item())
            
            # Track per-modality private stats
            for mod_name in ['heatmap', 'impedance', 'maxvalue']:
                mod_mu, mod_logvar = mod_stats[mod_name]
                mod_std = torch.exp(mod_logvar / 2)
                modality_stats[mod_name]['mu_mean'] += mod_mu.mean().item()
                modality_stats[mod_name]['mu_std'] += mod_mu.std().item()
                modality_stats[mod_name]['std_mean'] += mod_std.mean().item()
                modality_stats[mod_name]['std_std'] += mod_std.std().item()
            
            # Occupancy: track P(class=1) from Gumbel-Softmax
            _, occ_probs = mod_stats['occupancy']  # (B, occ_priv, 2)
            p1 = occ_probs[:, :, 1]  # P(bit=1) for each dim
            modality_stats['occupancy']['prob1_mean'] += p1.mean().item()
            modality_stats['occupancy']['prob1_std'] += p1.std().item()
        
        # Backward pass
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses (skip beta as it's constant)
        for key in total_losses:
            if key != 'beta':
                total_losses[key] += losses[key].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'recon': f"{losses['recon_loss'].item():.4f}",
            'kl': f"{losses['kl_loss'].item():.4f}"
        })
    
    # Average losses and stats (skip beta)
    num_batches = len(train_loader)
    for key in total_losses:
        if key != 'beta':
            total_losses[key] /= num_batches
    
    # Average mu and logvar stats
    mu_stats['mean'] /= num_batches
    mu_stats['std'] /= num_batches
    logvar_stats['mean'] /= num_batches
    logvar_stats['std'] /= num_batches
    
    # Average modality stats
    for mod_name in ['heatmap', 'impedance', 'maxvalue']:
        for key in modality_stats[mod_name]:
            modality_stats[mod_name][key] /= num_batches
    modality_stats['occupancy']['prob1_mean'] /= num_batches
    modality_stats['occupancy']['prob1_std'] /= num_batches
    
    # Add stats to total_losses for return
    total_losses['mu_mean'] = mu_stats['mean']
    total_losses['mu_std'] = mu_stats['std']
    total_losses['mu_min'] = mu_stats['min']
    total_losses['mu_max'] = mu_stats['max']
    total_losses['logvar_mean'] = logvar_stats['mean']
    total_losses['logvar_std'] = logvar_stats['std']
    total_losses['logvar_min'] = logvar_stats['min']
    total_losses['logvar_max'] = logvar_stats['max']
    total_losses['modality_stats'] = modality_stats
    
    return total_losses


def validate(model, val_loader, config, beta, temperature):
    """Validate the model with current beta and temperature"""
    model.eval()
    total_losses = {'total_loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0,
                   'kl_gaussian': 0.0, 'kl_gumbel': 0.0,
                   'heatmap_loss': 0.0, 'occupancy_loss': 0.0, 
                   'impedance_loss': 0.0,  'beta': beta}
    
    # Track mu and logvar statistics (Gaussian parts only)
    mu_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    logvar_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    std_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    
    # Track per-modality private stats
    modality_stats = {
        'heatmap':   {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0},
        'occupancy': {'prob1_mean': 0.0, 'prob1_std': 0.0},
        'impedance': {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0},
        'maxvalue':  {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0},
        'shared':    {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0}
    }
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            heatmap = batch['heatmap_norm'].to(config.device)
            occupancy = batch['occupancy'].to(config.device)
            impedance = batch['impedance'].to(config.device)
            maxvalue = batch['max_impedance_std'].to(config.device)
            
            # Ensure correct shapes
            if heatmap.dim() == 3:
                heatmap = heatmap.unsqueeze(1)
            if impedance.dim() == 1:
                impedance = impedance.unsqueeze(0)
            if maxvalue.dim() == 1:
                maxvalue = maxvalue.unsqueeze(1)
            
            # Forward pass
            recon_hm, recon_occ, recon_imp, recon_mv, mu_gaussian, logvar_gaussian, occ_logits, mod_stats = model(
                heatmap, occupancy, impedance, maxvalue, temperature=temperature
            )
            
            # Compute loss
            losses = vae_loss(recon_hm, recon_occ, recon_imp, recon_mv,
                            heatmap, occupancy, impedance, maxvalue,
                            mu_gaussian, logvar_gaussian, occ_logits, beta, config)
            
            # Track mu and logvar statistics (Gaussian parts only)
            mu_stats['mean'] += mu_gaussian.mean().item()
            mu_stats['std'] += mu_gaussian.std().item()
            mu_stats['min'] = min(mu_stats['min'], mu_gaussian.min().item())
            mu_stats['max'] = max(mu_stats['max'], mu_gaussian.max().item())
            
            logvar_stats['mean'] += logvar_gaussian.mean().item()
            logvar_stats['std'] += logvar_gaussian.std().item()
            logvar_stats['min'] = min(logvar_stats['min'], logvar_gaussian.min().item())
            logvar_stats['max'] = max(logvar_stats['max'], logvar_gaussian.max().item())
            
            std = torch.exp(logvar_gaussian / 2)
            std_stats['mean'] += std.mean().item()
            std_stats['std'] += std.std().item()
            std_stats['min'] = min(std_stats['min'], std.min().item())
            std_stats['max'] = max(std_stats['max'], std.max().item())
            
            # Track per-modality private stats
            for mod_name in ['heatmap', 'impedance', 'maxvalue']:
                mod_mu, mod_logvar = mod_stats[mod_name]
                mod_std = torch.exp(mod_logvar / 2)
                modality_stats[mod_name]['mu_mean'] += mod_mu.mean().item()
                modality_stats[mod_name]['mu_std'] += mod_mu.std().item()
                modality_stats[mod_name]['std_mean'] += mod_std.mean().item()
                modality_stats[mod_name]['std_std'] += mod_std.std().item()
            
            # Track shared latent stats (last shared_dim dims of mu_gaussian)
            shared_dim = model.shared_dim
            shared_mu = mu_gaussian[:, -shared_dim:]
            shared_logvar = logvar_gaussian[:, -shared_dim:]
            shared_std = torch.exp(shared_logvar / 2)
            modality_stats['shared']['mu_mean'] += shared_mu.mean().item()
            modality_stats['shared']['mu_std'] += shared_mu.std().item()
            modality_stats['shared']['std_mean'] += shared_std.mean().item()
            modality_stats['shared']['std_std'] += shared_std.std().item()
            
            # Occupancy: track P(class=1) from Gumbel-Softmax
            _, occ_probs = mod_stats['occupancy']
            p1 = occ_probs[:, :, 1]
            modality_stats['occupancy']['prob1_mean'] += p1.mean().item()
            modality_stats['occupancy']['prob1_std'] += p1.std().item()
            
            # Accumulate losses (skip beta as it's constant)
            for key in total_losses:
                if key != 'beta':
                    total_losses[key] += losses[key].item()
    
    # Average losses and stats (skip beta)
    num_batches = len(val_loader)
    for key in total_losses:
        if key != 'beta':
            total_losses[key] /= num_batches
    
    # Average mu and logvar stats
    mu_stats['mean'] /= num_batches
    mu_stats['std'] /= num_batches
    logvar_stats['mean'] /= num_batches
    logvar_stats['std'] /= num_batches
    std_stats['mean'] /= num_batches
    std_stats['std'] /= num_batches
    
    # Average modality stats
    for mod_name in modality_stats:
        for key in modality_stats[mod_name]:
            modality_stats[mod_name][key] /= num_batches
    
    # Add stats to total_losses for return
    total_losses['mu_mean'] = mu_stats['mean']
    total_losses['mu_std'] = mu_stats['std']
    total_losses['mu_min'] = mu_stats['min']
    total_losses['mu_max'] = mu_stats['max']
    total_losses['logvar_mean'] = logvar_stats['mean']
    total_losses['logvar_std'] = logvar_stats['std']
    total_losses['logvar_min'] = logvar_stats['min']
    total_losses['logvar_max'] = logvar_stats['max']
    total_losses['std_mean'] = std_stats['mean']
    total_losses['std_std'] = std_stats['std']
    total_losses['std_min'] = std_stats['min']
    total_losses['std_max'] = std_stats['max']
    
    # Add modality-specific stats
    total_losses['modality_stats'] = modality_stats
    
    return total_losses


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def train_vae():
    config = Config()
    
    # Create experiment directory
    exp_path = Path(config.experiment_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = exp_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)
    log_path = exp_path / "logs"
    log_path.mkdir(exist_ok=True)
    
    # Initialize logger
    logger = VAETrainingLogger(log_dir=str(log_path), checkpoint_dir=str(checkpoint_path))
    
    print("="*80)
    print("SIMPLE VAE TRAINING")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Experiment: {config.experiment_dir}")
    print(f"Logs: {log_path}")
    print(f"Checkpoints: {checkpoint_path}")
    print(f"Latent dim: {config.latent_dim}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    if config.use_beta_annealing:
        print(f"Beta Annealing: {config.beta_initial} → {config.beta_final} over epochs {config.beta_start_epoch}-{config.beta_end_epoch}")
    print("="*80)
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_data_loaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        normalize=False,  # Data already normalized
        train_split=config.train_split,
        seed=42
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = MultiInputVAE(latent_dim=config.latent_dim).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.resume_checkpoint:
        checkpoint_file = Path(config.resume_checkpoint)
        if checkpoint_file.exists():
            start_epoch, best_val_loss = load_checkpoint(
                config.resume_checkpoint, 
                model, 
                optimizer,
                config.device
            )
        else:
            print(f"\n⚠️  WARNING: Checkpoint not found: {config.resume_checkpoint}")
            print("Starting training from scratch...\n")
    else:
        print("\nStarting training from scratch...")
    
    # Training loop
    print(f"\nTraining from epoch {start_epoch + 1} to {config.num_epochs}...")
    print(f"Gumbel temperature: {config.gumbel_temp_start} → {config.gumbel_temp_end} over {config.gumbel_anneal_epochs} epochs")
    
    for epoch in range(start_epoch, config.num_epochs):
        # Compute annealed beta for this epoch
        beta = compute_beta(epoch, config)
        
        # Compute Gumbel-Softmax temperature (anneal from warm to cold)
        if epoch < config.gumbel_anneal_epochs:
            progress = epoch / max(config.gumbel_anneal_epochs, 1)
            temperature = config.gumbel_temp_start + progress * (config.gumbel_temp_end - config.gumbel_temp_start)
        else:
            temperature = config.gumbel_temp_end
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, config, epoch, beta, temperature)
        
        # Validate
        val_losses = validate(model, val_loader, config, beta, temperature)
        
        # Prepare loss dict for logger
        train_loss_dict = {
            'total_loss': train_losses['total_loss'],
            'recon_loss': train_losses['recon_loss'],
            'kl_loss': train_losses['kl_loss'],
            'kl_gaussian': train_losses['kl_gaussian'],
            'kl_gumbel': train_losses['kl_gumbel'],
            'heatmap_loss': train_losses['heatmap_loss'],
            'occupancy_loss': train_losses['occupancy_loss'],
            'impedance_loss': train_losses['impedance_loss']
        }
        
        # Log training losses
        logger.log_dict(epoch=epoch+1, loss_dict=train_loss_dict)
        
        # Print validation summary
        print(f"  Val   - Total: {val_losses['total_loss']:.4f}, "
              f"Recon: {val_losses['recon_loss']:.4f}, "
              f"KL: {val_losses['kl_loss']:.4f} "
              f"(gauss={val_losses['kl_gaussian']:.4f}, gumbel={val_losses['kl_gumbel']:.4f})")
        
        # Print latent space statistics
        v_ms = val_losses['modality_stats']
        print(f"  Latent - μ: {val_losses['mu_mean']:.4f}, σ: {val_losses['std_mean']:.4f} | "
              f"hm={v_ms['heatmap']['mu_mean']:.3f}/{v_ms['heatmap']['std_mean']:.3f}, "
              f"occ P(1)={v_ms['occupancy']['prob1_mean']:.3f}, "
              f"imp={v_ms['impedance']['mu_mean']:.3f}/{v_ms['impedance']['std_mean']:.3f}, "
              f"mv={v_ms['maxvalue']['mu_mean']:.3f}/{v_ms['maxvalue']['std_mean']:.3f}, "
              f"shared={v_ms['shared']['mu_mean']:.3f}/{v_ms['shared']['std_mean']:.3f}")
        
        # Build latent stats dict for inference sampling
        v_ms = val_losses['modality_stats']
        latent_stats = {
            'heatmap':   {'mu_mean': v_ms['heatmap']['mu_mean'],   'mu_std': v_ms['heatmap']['mu_std'],   'sigma_mean': v_ms['heatmap']['std_mean']},
            'impedance': {'mu_mean': v_ms['impedance']['mu_mean'], 'mu_std': v_ms['impedance']['mu_std'], 'sigma_mean': v_ms['impedance']['std_mean']},
            'maxvalue':  {'mu_mean': v_ms['maxvalue']['mu_mean'],  'mu_std': v_ms['maxvalue']['mu_std'],  'sigma_mean': v_ms['maxvalue']['std_mean']},
            'shared':    {'mu_mean': v_ms['shared']['mu_mean'],    'mu_std': v_ms['shared']['mu_std'],    'sigma_mean': v_ms['shared']['std_mean']},
        }

        # Save checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses['total_loss'],
                'val_loss': val_losses['total_loss'],
                'config': vars(config),
                'temperature': temperature,
                'latent_stats': latent_stats,
                'mu_mean': val_losses['mu_mean'],
                'mu_std': val_losses['mu_std'],
                'mu_min': val_losses['mu_min'],
                'mu_max': val_losses['mu_max'],
            }

            torch.save(checkpoint_dict, checkpoint_file)
            print(f"  Saved checkpoint: {checkpoint_file.name}")
        
        # Save best model
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            best_model_file = checkpoint_path / "best_model.pt"
            best_model_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': vars(config),
                'temperature': temperature,
                'latent_stats': latent_stats,
                'mu_mean': val_losses['mu_mean'],
                'mu_std': val_losses['mu_std'],
                'mu_min': val_losses['mu_min'],
                'mu_max': val_losses['mu_max'],
            }
            
            torch.save(best_model_dict, best_model_file)
            print(f"  ✓ New best model! Val loss: {best_val_loss:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {checkpoint_path / 'best_model.pt'}")
    
    # Generate training plots
    print("\nGenerating training plots...")
    logger.plot()
    logger.plot_loss_components()
    
    # Print training statistics
    logger.print_statistics()


if __name__ == "__main__":
    train_vae()
