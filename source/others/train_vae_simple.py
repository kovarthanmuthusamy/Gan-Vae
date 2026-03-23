"""
Simple training script for Multi-Input VAE (vae_multi_input_simple.py)
Streamlined version without complex features
"""

import sys
import csv
import json
import random
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
    latent_dim: int = 132  # Total latent dim (private + shared)
    
    # Training
    num_epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1e-4
    
    # Loss weights
    heatmap_weight: float = 1.0
    occupancy_weight: float = 1.0
    impedance_weight: float = 1.0

    # Cross-modal reconstruction weight (trains shared space for cross-modal generation)
    # Each batch: randomly pick one source modality, encode it alone, decode all outputs.
    # Set to 0.0 to disable (reverts to standard ELBO-only training).
    cross_modal_weight: float = 0.5

    # Modality dropout: probability of dropping each modality from the shared PoE during training.
    # Forces each single modality to independently populate the shared space (enables cross-modal gen).
    modality_dropout: float = 0.3
    
    # Gumbel-Softmax temperature annealing for occupancy
    gumbel_temp_start: float = 1.0    # Start warm (soft samples)
    gumbel_temp_end: float = 0.1      # Anneal to cold (hard samples)
    gumbel_anneal_epochs: int = 150   # Anneal over this many epochs (slow to preserve gradients)
    
    
    # Free bits: minimum KL per latent dimension (prevents posterior collapse as β rises)
    free_bits: float = 0.5  # nats; dims below this KL are not penalised

    # Beta-VAE Annealing (KL weight annealing)
    use_beta_annealing: bool = True  # Set to True to enable beta annealing
    beta_start_epoch: int = 0        # Start annealing from epoch 0
    beta_end_epoch: int = 200        # Ramp up over 200 epochs
    beta_initial: float = 0.0        # Start with 0 KL weight
    beta_final: float = 0.01        # Ramp to final KL weight — enough to pull μ toward 0 for valid N(0,1) sampling
    
    # Data
    data_dir: str = "datasets/data_norm"
    train_split: float = 0.9
    num_workers: int = 4
    
    # Background sentinel (loaded from normalization stats at runtime)
    background_value: float = -3.6228  # default; overridden in train_vae()
    
    # Checkpoints

    experiment_dir: str = "experiments/exp021"  # Directory to save checkpoints and logs
    checkpoint_interval: int = 10
    resume_checkpoint: str = ""  
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



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
def vae_loss(recon_heatmap, recon_occupancy, recon_impedance,
             target_heatmap, target_occupancy, target_impedance,
             mu_gaussian, logvar_gaussian, occ_logits, beta, config):
    """
    Hybrid VAE loss: Reconstruction + Gaussian KL + Gumbel KL
    
    Args:
        mu_gaussian, logvar_gaussian: Gaussian parts (heatmap, impedance privates + shared)
        occ_logits: (B, occ_priv, 2) Gumbel-Softmax logits for occupancy
        beta: Annealed KL weight
    """
    # Heatmap loss (foreground-masked: exclude background sentinel pixels)
    # Sentinel = config.background_value (z-score of log(1+0) ≈ -3.62); foreground z_min ≈ -2.12
    fg_mask = (target_heatmap > config.background_value + 0.5).float()  # 1=foreground, 0=background
    huber_elem = F.huber_loss(recon_heatmap, target_heatmap, delta=3.0, reduction='none')
    fg_count = fg_mask.sum().clamp(min=1.0)
    loss_heatmap = (huber_elem * fg_mask).sum() / fg_count

    # Occupancy loss: binary cross-entropy
    loss_occupancy = F.binary_cross_entropy(
        recon_occupancy.clamp(1e-7, 1 - 1e-7),
        target_occupancy.clamp(0.0, 1.0),
        reduction='mean'
    )

    loss_impedance = F.huber_loss(recon_impedance, target_impedance, delta=5.0, reduction='mean')

    # Total reconstruction loss (weighted)
    recon_loss = (config.heatmap_weight * loss_heatmap +
                  config.occupancy_weight * loss_occupancy +
                  config.impedance_weight * loss_impedance)
    
    # === Gaussian KL divergence ===
    # Per-dim KL: 0.5 * (μ² + σ² - log(σ²) - 1)
    kl_per_dim = -0.5 * (1 + logvar_gaussian - mu_gaussian.pow(2) - logvar_gaussian.exp())  # (B, D)
    # Free-bits: don't penalise dims whose KL is already below the floor
    if config.free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=config.free_bits)
    kl_gaussian = kl_per_dim.mean()  # mean over batch and dims
    
    # === Binary Concrete KL divergence (for occupancy) ===
    # KL(Bernoulli(q) || Bernoulli(0.5)) per dim
    # = q * log(q/0.5) + (1-q) * log((1-q)/0.5)
    # = q*log(q) + (1-q)*log(1-q) + log(2)  [since prior is 0.5]
    occ_probs = torch.sigmoid(occ_logits)  # (B, occ_priv)
    occ_probs = occ_probs.clamp(1e-6, 1 - 1e-6)
    kl_bernoulli = (occ_probs * torch.log(occ_probs / 0.5) +
                    (1 - occ_probs) * torch.log((1 - occ_probs) / 0.5))  # (B, occ_priv)
    kl_bernoulli = kl_bernoulli.mean()  # mean over dims and batch
    
    # Total KL
    kl_loss = kl_gaussian + kl_bernoulli
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'kl_gaussian': kl_gaussian,
        'kl_bernoulli': kl_bernoulli,
        'heatmap_loss': loss_heatmap,
        'occupancy_loss': loss_occupancy,
        'impedance_loss': loss_impedance
    }


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, train_loader, optimizer, config, epoch, beta, temperature):
    """Train for one epoch with annealed beta and Binary Concrete temperature"""
    model.train()
    total_losses = {'total_loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0,
                   'kl_gaussian': 0.0, 'kl_bernoulli': 0.0,
                   'heatmap_loss': 0.0, 'occupancy_loss': 0.0, 
                   'impedance_loss': 0.0,  'beta': beta}
    
    # Track mu and logvar statistics (Gaussian parts only)
    mu_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    logvar_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    std_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    
    # Track per-modality private stats
    modality_stats = {
        'heatmap':   {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0},
        'occupancy': {'prob1_mean': 0.0, 'prob1_std': 0.0,
                     'logit_mag': 0.0, 'entropy': 0.0, 'sharp_frac': 0.0},  # Gumbel stats
        'impedance': {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0},
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} (β={beta:.5f}, τ={temperature:.3f})")
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        heatmap = batch['heatmap_norm'].to(config.device)
        occupancy = batch['occupancy'].to(config.device)
        impedance = batch['impedance'].to(config.device)
        
        # Ensure correct shapes
        if heatmap.dim() == 3:
            heatmap = heatmap.unsqueeze(1)
        if impedance.dim() == 1:
            impedance = impedance.unsqueeze(0)
        
        # Replace background sentinel with 0 (z-score mean) for encoder input.
        # Sentinel (-3.62) is ~20σ outside foreground range and pollutes encoder activations.
        # Loss masking uses the original heatmap, so reconstruction targets are unaffected.
        heatmap_enc = heatmap.masked_fill(heatmap < config.background_value + 0.5, 0.0)

        # Forward pass
        optimizer.zero_grad()
        recon_hm, recon_occ, recon_imp, mu_gaussian, logvar_gaussian, occ_logits, mod_stats = model(
            heatmap_enc, occupancy, impedance, temperature=temperature
        )
        
        # Compute loss with annealed beta (hybrid: Gaussian KL + Gumbel KL)
        losses = vae_loss(recon_hm, recon_occ, recon_imp,
                         heatmap, occupancy, impedance,
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
            for mod_name in ['heatmap', 'impedance']:
                mod_mu, mod_logvar = mod_stats[mod_name]
                mod_std = torch.exp(mod_logvar / 2)
                modality_stats[mod_name]['mu_mean'] += mod_mu.mean().item()
                modality_stats[mod_name]['mu_std'] += mod_mu.std().item()
                modality_stats[mod_name]['std_mean'] += mod_std.mean().item()
                modality_stats[mod_name]['std_std'] += mod_std.std().item()
            
            # Occupancy: track P(bit=1) from Binary Concrete
            occ_logits_raw, occ_probs = mod_stats['occupancy']  # logits: (B, occ_priv), probs: (B, occ_priv)
            p1 = occ_probs  # P(bit=1) = sigmoid(logit) per dim
            modality_stats['occupancy']['prob1_mean'] += p1.mean().item()
            modality_stats['occupancy']['prob1_std'] += p1.std().item()
            # Logit magnitude: |logit| — how confident the encoder is
            modality_stats['occupancy']['logit_mag'] += occ_logits_raw.abs().mean().item()
            # Binary entropy: -[p*log(p) + (1-p)*log(1-p)], max=ln(2)≈0.693
            ent = -(p1 * torch.log(p1 + 1e-8) + (1 - p1) * torch.log(1 - p1 + 1e-8))  # (B, occ_priv)
            modality_stats['occupancy']['entropy'] += ent.mean().item()
            # Sharp fraction: dims where P(1) > 0.8 or P(1) < 0.2
            sharp = ((p1 > 0.8) | (p1 < 0.2)).float().mean()
            modality_stats['occupancy']['sharp_frac'] += sharp.item()
        
        # Cross-modal reconstruction: encode from one random source modality, decode all outputs.
        # Provides a direct gradient signal forcing the shared space to carry cross-modal info.
        # This runs outside torch.no_grad() so gradients flow back through the source encoder.
        if config.cross_modal_weight > 0:
            source = random.choice(['heatmap', 'impedance'])
            z_cm = model.encode_cross_modal(
                source, heatmap=heatmap_enc, impedance=impedance, temperature=temperature
            )
            recon_hm_cm, recon_occ_cm, recon_imp_cm = model.decode(z_cm)
            fg_mask_cm = (heatmap > config.background_value + 0.5).float()
            loss_hm_cm  = (F.huber_loss(recon_hm_cm, heatmap, delta=1.0, reduction='none') * fg_mask_cm).sum() / fg_mask_cm.sum().clamp(min=1.0)
            loss_occ_cm = F.binary_cross_entropy(recon_occ_cm.clamp(1e-7, 1-1e-7), occupancy.clamp(0, 1), reduction='mean')
            loss_imp_cm = F.huber_loss(recon_imp_cm, impedance, delta=5.0, reduction='mean')
            loss_cm = (config.heatmap_weight * loss_hm_cm +
                       config.occupancy_weight * loss_occ_cm +
                       config.impedance_weight * loss_imp_cm)
            losses['total_loss'] = losses['total_loss'] + config.cross_modal_weight * loss_cm

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
    for mod_name in ['heatmap', 'impedance']:
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
                   'kl_gaussian': 0.0, 'kl_bernoulli': 0.0,
                   'heatmap_loss': 0.0, 'occupancy_loss': 0.0, 
                   'impedance_loss': 0.0,  'beta': beta}
    
    # Track mu and logvar statistics (Gaussian parts only)
    mu_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    logvar_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    std_stats = {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')}
    
    # Track per-modality private stats
    modality_stats = {
        'heatmap':   {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0},
        'occupancy': {'prob1_mean': 0.0, 'prob1_std': 0.0,
                     'logit_mag': 0.0, 'entropy': 0.0, 'sharp_frac': 0.0},
        'impedance': {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0},
        'shared':    {'mu_mean': 0.0, 'mu_std': 0.0, 'std_mean': 0.0, 'std_std': 0.0}
    }
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            heatmap = batch['heatmap_norm'].to(config.device)
            occupancy = batch['occupancy'].to(config.device)
            impedance = batch['impedance'].to(config.device)
            
            # Ensure correct shapes
            if heatmap.dim() == 3:
                heatmap = heatmap.unsqueeze(1)
            if impedance.dim() == 1:
                impedance = impedance.unsqueeze(0)
            
            # Replace background sentinel with 0 for encoder input (same as train)
            heatmap_enc = heatmap.masked_fill(heatmap < config.background_value + 0.5, 0.0)

            # Forward pass
            recon_hm, recon_occ, recon_imp, mu_gaussian, logvar_gaussian, occ_logits, mod_stats = model(
                heatmap_enc, occupancy, impedance, temperature=temperature
            )
            
            # Compute loss
            losses = vae_loss(recon_hm, recon_occ, recon_imp,
                            heatmap, occupancy, impedance,
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
            for mod_name in ['heatmap', 'impedance']:
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
            
            # Occupancy: track P(bit=1) from Binary Concrete
            occ_logits_raw, occ_probs = mod_stats['occupancy']
            p1 = occ_probs  # (B, occ_priv)
            modality_stats['occupancy']['prob1_mean'] += p1.mean().item()
            modality_stats['occupancy']['prob1_std'] += p1.std().item()
            # Logit magnitude
            modality_stats['occupancy']['logit_mag'] += occ_logits_raw.abs().mean().item()
            # Binary entropy
            ent = -(p1 * torch.log(p1 + 1e-8) + (1 - p1) * torch.log(1 - p1 + 1e-8))
            modality_stats['occupancy']['entropy'] += ent.mean().item()
            # Sharp fraction
            sharp = ((p1 > 0.8) | (p1 < 0.2)).float().mean()
            modality_stats['occupancy']['sharp_frac'] += sharp.item()
            
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
    latent_stats_csv = log_path / "latent_stats.csv"
    
    # Initialize logger
    logger = VAETrainingLogger(log_dir=str(log_path), checkpoint_dir=str(checkpoint_path))
    
    print("="*80)
    print("SIMPLE VAE TRAINING")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Experiment: {config.experiment_dir}")
    print(f"Latent dim: {config.latent_dim}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    if config.use_beta_annealing:
        print(f"Beta: {config.beta_initial} → {config.beta_final} over epochs {config.beta_start_epoch}-{config.beta_end_epoch}")
    print(f"Gumbel temp: {config.gumbel_temp_start} → {config.gumbel_temp_end} over {config.gumbel_anneal_epochs} epochs")
    print("="*80)
    
    # Load background sentinel from normalization stats
    stats_path = Path(config.data_dir) / "normalization_stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            norm_stats = json.load(f)
        config.background_value = norm_stats["background_value"]
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        normalize=False,  # Data already normalized
        train_split=config.train_split,
        seed=42
    )
    
    # Create model
    model = MultiInputVAE(latent_dim=config.latent_dim, modality_dropout=config.modality_dropout).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
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
            print(f"\n⚠️  Checkpoint not found: {config.resume_checkpoint}, starting from scratch")
    
    # Training loop
    print(f"\nTraining epochs {start_epoch + 1}–{config.num_epochs}...")
    
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
            'kl_bernoulli': train_losses['kl_bernoulli'],
            'heatmap_loss': train_losses['heatmap_loss'],
            'occupancy_loss': train_losses['occupancy_loss'],
            'impedance_loss': train_losses['impedance_loss']
        }
        
        # Log training losses
        logger.log_dict(epoch=epoch+1, loss_dict=train_loss_dict)
        
        # Print validation summary
        print(f"  Val - Loss: {val_losses['total_loss']:.4f} "
              f"(recon={val_losses['recon_loss']:.4f}, "
              f"kl={val_losses['kl_loss']:.4f})")
        
        # Print per-modality latent stats
        v_ms = val_losses['modality_stats']
        print(f"  Latent μ/σ - "
              f"hm: {v_ms['heatmap']['mu_mean']:.3f}/{v_ms['heatmap']['std_mean']:.3f}, "
              f"occ P(1): {v_ms['occupancy']['prob1_mean']:.3f}, "
              f"imp: {v_ms['impedance']['mu_mean']:.3f}/{v_ms['impedance']['std_mean']:.3f}, "
              f"shared: {v_ms['shared']['mu_mean']:.3f}/{v_ms['shared']['std_mean']:.3f}")
        # Print Gumbel-Softmax diagnostics
        occ_s = v_ms['occupancy']
        print(f"  Gumbel - logit_mag: {occ_s.get('logit_mag', 0):.3f}, "
              f"entropy: {occ_s.get('entropy', 0):.3f}/0.693, "
              f"sharp%%: {occ_s.get('sharp_frac', 0)*100:.1f}%")

        # Log latent statistics to CSV every 10 epochs
        if (epoch + 1) % 10 == 0:
            write_header = not latent_stats_csv.exists()
            with open(latent_stats_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'epoch', 'beta', 'temperature',
                        'hm_mu_mean', 'hm_mu_std', 'hm_sigma_mean', 'hm_sigma_std',
                        'imp_mu_mean', 'imp_mu_std', 'imp_sigma_mean', 'imp_sigma_std',
                        'shared_mu_mean', 'shared_mu_std', 'shared_sigma_mean', 'shared_sigma_std',
                        'occ_prob1_mean', 'occ_logit_mag', 'occ_entropy', 'occ_sharp_pct',
                        'kl_gaussian', 'kl_bernoulli', 'kl_total',
                        'val_recon', 'val_total'
                    ])
                writer.writerow([
                    epoch + 1, f"{beta:.6f}", f"{temperature:.4f}",
                    f"{v_ms['heatmap']['mu_mean']:.4f}",   f"{v_ms['heatmap']['mu_std']:.4f}",
                    f"{v_ms['heatmap']['std_mean']:.4f}",  f"{v_ms['heatmap']['std_std']:.4f}",
                    f"{v_ms['impedance']['mu_mean']:.4f}", f"{v_ms['impedance']['mu_std']:.4f}",
                    f"{v_ms['impedance']['std_mean']:.4f}",f"{v_ms['impedance']['std_std']:.4f}",
                    f"{v_ms['shared']['mu_mean']:.4f}",    f"{v_ms['shared']['mu_std']:.4f}",
                    f"{v_ms['shared']['std_mean']:.4f}",   f"{v_ms['shared']['std_std']:.4f}",
                    f"{v_ms['occupancy']['prob1_mean']:.4f}",
                    f"{v_ms['occupancy'].get('logit_mag', 0):.4f}",
                    f"{v_ms['occupancy'].get('entropy', 0):.4f}",
                    f"{v_ms['occupancy'].get('sharp_frac', 0)*100:.2f}",
                    f"{val_losses['kl_gaussian']:.4f}",
                    f"{val_losses['kl_bernoulli']:.4f}",
                    f"{val_losses['kl_loss']:.4f}",
                    f"{val_losses['recon_loss']:.4f}",
                    f"{val_losses['total_loss']:.4f}",
                ])

        # Build latent stats dict for inference sampling
        v_ms = val_losses['modality_stats']
        latent_stats = {
            'heatmap':   {'mu_mean': v_ms['heatmap']['mu_mean'],   'mu_std': v_ms['heatmap']['mu_std'],   'sigma_mean': v_ms['heatmap']['std_mean']},
            'impedance': {'mu_mean': v_ms['impedance']['mu_mean'], 'mu_std': v_ms['impedance']['mu_std'], 'sigma_mean': v_ms['impedance']['std_mean']},
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
    print(f"Checkpoints: {checkpoint_path}")
    
    # Generate training plots
    logger.plot()
    logger.plot_loss_components()
    logger.print_statistics()


if __name__ == "__main__":
    train_vae()
