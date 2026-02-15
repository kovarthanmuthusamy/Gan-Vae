"""
Loss function for Multi-Input Variational Autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Dice Loss for binary segmentation (occupancy maps)
    
    Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
    Dice loss: 1 - Dice coefficient
    
    Args:
        pred_logits: Predicted logits (B, 1, H, W) - before sigmoid
        target: Ground truth binary mask (B, 1, H, W) - values in [0, 1]
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss value
    """
    # Apply sigmoid to convert logits to probabilities
    pred_probs = torch.sigmoid(pred_logits)
    
    # Flatten spatial dimensions for batch-wise computation
    pred_flat = pred_probs.view(pred_probs.size(0), -1)  # (B, H*W)
    target_flat = target.view(target.size(0), -1)        # (B, H*W)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)  # (B,)
    pred_sum = pred_flat.sum(dim=1)                      # (B,)
    target_sum = target_flat.sum(dim=1)                  # (B,)
    
    # Dice coefficient per sample
    dice_coeff = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Dice loss (1 - Dice coefficient), averaged over batch
    return 1.0 - dice_coeff.mean()


def focal_loss(pred_logits: torch.Tensor, target: torch.Tensor, 
               alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean') -> torch.Tensor:
    """
    Focal Loss for handling class imbalance in sparse occupancy grids.
    
    Focal loss down-weights "easy" examples and focuses on "hard" examples.
    Perfect for sparse occupancy maps where most pixels are background (0).
    
    Formula: FL = -α(1-p_t)^γ * log(p_t)
    where p_t = p if y=1 else (1-p)
    
    Args:
        pred_logits: Predicted logits (B, 1, H, W)
        target: Ground truth binary mask (B, 1, H, W) - values in [0, 1]  
        alpha: Weighting factor for rare class (default 0.25)
        gamma: Focusing parameter (default 2.0, higher = more focus on hard examples)
        reduction: 'mean' or 'sum'
    
    Returns:
        Focal loss value
    """
    # Get probabilities
    pred_probs = torch.sigmoid(pred_logits)
    
    # Compute p_t: probability of the true class
    p_t = pred_probs * target + (1 - pred_probs) * (1 - target)
    
    # Compute alpha_t: alpha for positive class, (1-alpha) for negative class
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t).pow(gamma)
    
    # Compute focal loss: -alpha_t * focal_weight * log(p_t)
    # Add small epsilon to prevent log(0)
    focal_loss_val = -alpha_t * focal_weight * torch.log(p_t + 1e-8)
    
    if reduction == 'mean':
        return focal_loss_val.mean()
    elif reduction == 'sum':
        return focal_loss_val.sum()
    else:
        return focal_loss_val


def cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine Similarity Loss for impedance vectors (shape & direction)
    
    Measures the angle between predicted and target vectors, focusing on pattern/shape
    rather than absolute magnitude. This is useful when the relative changes in the 
    impedance vector are more important than absolute values.
    
    Formula: L_cos = 1 - (y · ŷ) / (||y|| ||ŷ||)
    
    Args:
        pred: Predicted impedance vector (B, D) where D is impedance dimension
        target: Ground truth impedance vector (B, D)
        eps: Small epsilon for numerical stability
    
    Returns:
        Cosine similarity loss value (0 = perfect match, 2 = opposite direction)
    """
    # Compute cosine similarity: (y · ŷ) / (||y|| ||ŷ||)
    # dim=1 computes similarity along the feature dimension for each sample
    cos_sim = F.cosine_similarity(pred, target, dim=1, eps=eps)  # (B,)
    
    # Cosine loss: 1 - cosine_similarity
    # Range: [0, 2] where 0 = identical, 1 = orthogonal, 2 = opposite
    cos_loss = 1.0 - cos_sim.mean()
    
    return cos_loss


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, 
              window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) Loss for heatmaps
    
    SSIM measures perceptual similarity by comparing luminance, contrast, and structure.
    Better than MSE for image quality as it aligns with human visual perception.
    
    Formula: SSIM(x,y) = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
    SSIM Loss = 1 - SSIM
    
    Args:
        pred: Predicted heatmap (B, C, H, W)
        target: Ground truth heatmap (B, C, H, W)
        window_size: Size of Gaussian window for local comparison
        size_average: Whether to average across all pixels
    
    Returns:
        SSIM loss value (0 = perfect match, 1 = completely different)
    """
    C1 = 0.01 ** 2  # Constant for luminance stability
    C2 = 0.03 ** 2  # Constant for contrast stability
    
    # Create Gaussian window
    channel = pred.size(1)
    sigma = 1.5
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    # Create 2D Gaussian kernel
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(pred.device)
    
    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return 1.0 - ssim_map.mean()  # SSIM loss
    else:
        return 1.0 - ssim_map.mean(dim=(1, 2, 3))  # Per-sample SSIM loss


class VAELoss(nn.Module):
    """
    Combined loss function for multi-output VAE with Uncertainty-Based Weighting (Kendall et al.)
    
    This implementation combines proper KL-reconstruction scaling with learnable uncertainty parameters
    for automatic multi-task balancing. Instead of fixed weights, each modality has a learnable
    log-variance parameter σ that represents homoscedastic uncertainty.
    
    Key Features:
    - Multi-modal reconstruction losses (heatmap, occupancy, impedance)
    - Learnable uncertainty-based weighting: L_i = (1/(2*σ_i^2)) * MSE_i + (1/2) * log(σ_i^2)
    - KL divergence loss scaled proportionally to reconstruction space dimensionality
    - Automatic down-weighting of "noisy" or "easy" tasks
    - Forces optimizer focus on modalities with high reconstruction error
    
    Uncertainty Weighting Benefits:
    - Eliminates need for manual weight tuning
    - Adapts automatically during training
    - Prevents any single modality from dominating
    - More robust to different modality scales
    
    Usage Notes:
    - Uncertainty parameters (log_sigma) are learned alongside model weights
    - Higher uncertainty (σ) → lower weight for that modality loss
    - Regularization term prevents σ from growing infinitely
    - Initial values matter: start with small log_sigma (around -1 to 0)
    """
    
    def __init__(self, 
                 # Uncertainty-based weights (learnable parameters)
                 init_log_sigma_heatmap: float = -0.5,
                 init_log_sigma_occupancy: float = -0.5, 
                 init_log_sigma_impedance: float = -0.5,
                 # KL weight (still fixed)
                 kl_weight: float = 0.001,
                 # Occupancy loss configuration
                 occupancy_loss_type: str = 'weighted_bce',  # 'bce', 'weighted_bce', 'focal', 'combo'
                 occupancy_pos_weight: float = 15.0,  # For weighted BCE: emphasize positive class
                 focal_alpha: float = 0.25,  # For focal loss
                 focal_gamma: float = 2.0,   # For focal loss  
                 static_occupancy_weight: Optional[float] = None,  # If set, override uncertainty for first N epochs
                 static_weight_epochs: int = 50,  # Number of epochs to use static weight
                 # Sub-loss weights within modalities (still fixed for now)
                 occupancy_bce_weight: float = 0.5,
                 occupancy_dice_weight: float = 0.5,
                 impedance_cosine_weight: float = 0.5,
                 impedance_mse_weight: float = 0.5):
        super(VAELoss, self).__init__()
        
        # Learnable uncertainty parameters (log variance)
        # Using nn.Parameter makes them trainable
        self.log_sigma_heatmap = nn.Parameter(torch.tensor(init_log_sigma_heatmap, dtype=torch.float32))
        self.log_sigma_occupancy = nn.Parameter(torch.tensor(init_log_sigma_occupancy, dtype=torch.float32))
        self.log_sigma_impedance = nn.Parameter(torch.tensor(init_log_sigma_impedance, dtype=torch.float32))
        
        # Fixed parameters
        self.kl_weight = kl_weight
        self.occupancy_bce_weight = occupancy_bce_weight
        self.occupancy_dice_weight = occupancy_dice_weight
        self.impedance_cosine_weight = impedance_cosine_weight
        self.impedance_mse_weight = impedance_mse_weight
        
        # 🎯 FIX 1: Occupancy loss configuration
        self.occupancy_loss_type = occupancy_loss_type
        self.occupancy_pos_weight = occupancy_pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 🎯 FIX 2: Static occupancy weight override
        self.static_occupancy_weight = static_occupancy_weight
        self.static_weight_epochs = static_weight_epochs
        self.current_epoch = 0  # Track current epoch for static weight logic
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
        # 🎯 FIX 1: Weighted BCE with pos_weight for sparse occupancy grids
        # Register pos_weight as a buffer so it moves with the model to the correct device
        pos_weight_tensor = torch.tensor([self.occupancy_pos_weight], dtype=torch.float32)
        self.register_buffer('pos_weight', pos_weight_tensor)
        self.weighted_bce_loss = None  # Will be created lazily with correct device
        
        self.mae_loss = nn.L1Loss(reduction='mean')
        
        print(f"🎯 Initialized ENHANCED VAE Loss with Occupancy Fixes:")
        print(f"  Log-sigma heatmap: {init_log_sigma_heatmap:.3f} (σ={torch.exp(torch.tensor(init_log_sigma_heatmap)):.3f})")
        print(f"  Log-sigma occupancy: {init_log_sigma_occupancy:.3f} (σ={torch.exp(torch.tensor(init_log_sigma_occupancy)):.3f})")
        print(f"  Log-sigma impedance: {init_log_sigma_impedance:.3f} (σ={torch.exp(torch.tensor(init_log_sigma_impedance)):.3f})")
        print(f"  🔥 Occupancy loss type: {occupancy_loss_type}")
        if occupancy_loss_type in ['weighted_bce', 'combo']:
            print(f"  ⚖️  Pos weight: {occupancy_pos_weight}x (makes missing '1's {occupancy_pos_weight}x more painful)")
        if occupancy_loss_type in ['dice']:
            print(f"  🎲 Using Dice Loss (overlap-based, no pos_weight needed)")
        if occupancy_loss_type in ['focal', 'combo', 'dice_focal']:
            print(f"  🎯 Focal: α={focal_alpha}, γ={focal_gamma} (focuses on hard examples)")
        if static_occupancy_weight is not None:
            print(f"  🔒 Static occupancy weight: {static_occupancy_weight} for first {static_weight_epochs} epochs")
    def set_current_epoch(self, epoch: int):
        """Update current epoch for static weight logic"""
        self.current_epoch = epoch
        
    def compute_occupancy_loss(self, occupancy_logits: torch.Tensor, occupancy_target: torch.Tensor) -> torch.Tensor:
        """
        🎯 Compute occupancy loss with various strategies for sparse grids
        
        Args:
            occupancy_logits: Predicted logits (B, 1, H, W)
            occupancy_target: Ground truth binary mask (B, 1, H, W)
        
        Returns:
            Occupancy loss value
        """
        if self.occupancy_loss_type == 'bce':
            # Standard BCE (original, tends to predict all zeros)
            return self.bce_loss(occupancy_logits, occupancy_target)
            
        elif self.occupancy_loss_type == 'weighted_bce':
            # 🎯 FIX 1: Weighted BCE - emphasizes positive class (requires device-specific pos_weight)
            pos_weight_tensor = self.pos_weight if isinstance(self.pos_weight, torch.Tensor) else None
            return F.binary_cross_entropy_with_logits(
                occupancy_logits, occupancy_target, pos_weight=pos_weight_tensor, reduction='mean'
            )
            
        elif self.occupancy_loss_type == 'dice':
            # 🎯 Dice Loss - designed for sparse grids, focuses on overlap/shape
            return dice_loss(occupancy_logits, occupancy_target)
            
        elif self.occupancy_loss_type == 'focal':
            # 🎯 Focal loss - focuses on hard examples, no pos_weight needed
            return focal_loss(occupancy_logits, occupancy_target, 
                            alpha=self.focal_alpha, gamma=self.focal_gamma)
            
        elif self.occupancy_loss_type == 'dice_focal':
            # 🎯 Dice + Focal combo - best for sparse grids, no pos_weight needed
            dice = dice_loss(occupancy_logits, occupancy_target)
            focal = focal_loss(occupancy_logits, occupancy_target, 
                             alpha=self.focal_alpha, gamma=self.focal_gamma)
            return 0.5 * dice + 0.5 * focal
            
        elif self.occupancy_loss_type == 'combo':
            # Legacy combo: Weighted BCE + Focal loss
            pos_weight_tensor = self.pos_weight if isinstance(self.pos_weight, torch.Tensor) else None
            weighted_bce = F.binary_cross_entropy_with_logits(
                occupancy_logits, occupancy_target, pos_weight=pos_weight_tensor, reduction='mean'
            )
            focal = focal_loss(occupancy_logits, occupancy_target, 
                             alpha=self.focal_alpha, gamma=self.focal_gamma)
            return 0.7 * weighted_bce + 0.3 * focal
            
        elif self.occupancy_loss_type == 'triple':
            # Triple combo: Dice + Focal + Weighted BCE
            dice = dice_loss(occupancy_logits, occupancy_target)
            focal = focal_loss(occupancy_logits, occupancy_target, 
                             alpha=self.focal_alpha, gamma=self.focal_gamma)
            pos_weight_tensor = self.pos_weight if isinstance(self.pos_weight, torch.Tensor) else None
            weighted_bce = F.binary_cross_entropy_with_logits(
                occupancy_logits, occupancy_target, pos_weight=pos_weight_tensor, reduction='mean'
            )
            return 1.0 * dice + 1.0 * focal + 0.5 * weighted_bce
            
        else:
            raise ValueError(f"Unknown occupancy_loss_type: {self.occupancy_loss_type}")

    def forward(self, outputs: Dict[str, torch.Tensor],
                heatmap_target: torch.Tensor,
                occupancy_target: torch.Tensor,
                impedance_target: torch.Tensor,
                kl_weight_multiplier: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Calculate VAE loss with uncertainty-based weighting and KL scaling
        
        The loss for each modality follows: L_i = (1/(2*σ_i^2)) * MSE_i + (1/2) * log(σ_i^2)
        where σ_i^2 = exp(log_sigma_i) is the learned uncertainty for modality i
        
        Args:
            outputs: Dictionary from VAE forward pass
            heatmap_target: Target heatmap (batch_size, 2, 64, 64)
            occupancy_target: Target occupancy (batch_size, 1, 7, 8)
            impedance_target: Target impedance (batch_size, 231)
            kl_weight_multiplier: Multiplier for KL weight (for annealing)
        
        Returns:
            Dictionary with loss components including uncertainty terms
        """
        mu = outputs['mu']
        logvar = outputs['logvar']
        heatmap_recon = outputs['heatmap_recon']
        occupancy_recon = outputs['occupancy_recon']
        occupancy_logits = outputs.get('occupancy_logits', occupancy_recon)
        impedance_recon = outputs['impedance_recon']
        
        # Check for NaN or Inf in outputs
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            raise ValueError("NaN or Inf detected in mu")
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            raise ValueError("NaN or Inf detected in logvar")
        if torch.isnan(heatmap_recon).any() or torch.isinf(heatmap_recon).any():
            raise ValueError("NaN or Inf detected in heatmap_recon")
        if torch.isnan(occupancy_logits).any() or torch.isinf(occupancy_logits).any():
            raise ValueError("NaN or Inf detected in occupancy_logits")
        if torch.isnan(impedance_recon).any() or torch.isinf(impedance_recon).any():
            raise ValueError("NaN or Inf detected in impedance_recon")
        
        # Compute individual reconstruction losses (raw, before uncertainty weighting)
        heatmap_loss_raw = self.mae_loss(heatmap_recon, heatmap_target) 
        
        # 🎯 Use enhanced occupancy loss computation
        occupancy_loss_raw = self.compute_occupancy_loss(occupancy_logits, occupancy_target)
        
        impedance_mse = self.mae_loss(impedance_recon, impedance_target)
        impedance_loss_raw = impedance_mse
        
        # Get current uncertainty values (σ^2 = exp(log_sigma))
        sigma2_heatmap = torch.exp(self.log_sigma_heatmap)
        sigma2_occupancy = torch.exp(self.log_sigma_occupancy) 
        sigma2_impedance = torch.exp(self.log_sigma_impedance)
        
        # Apply uncertainty-based weighting: L_i = (1/(2*σ_i^2)) * MSE_i + (1/2) * log(σ_i^2)
        heatmap_loss = (1.0 / (2.0 * sigma2_heatmap)) * heatmap_loss_raw + 0.5 * self.log_sigma_heatmap
        
        # 🎯 FIX 2: Static occupancy weight override for early epochs
        if (self.static_occupancy_weight is not None and 
            self.current_epoch < self.static_weight_epochs):
            # Use static weight instead of uncertainty-based weighting
            occupancy_loss = self.static_occupancy_weight * occupancy_loss_raw
        else:
            # Use normal uncertainty-based weighting
            occupancy_loss = (1.0 / (2.0 * sigma2_occupancy)) * occupancy_loss_raw + 0.5 * self.log_sigma_occupancy
        
        impedance_loss = (1.0 / (2.0 * sigma2_impedance)) * impedance_loss_raw + 0.5 * self.log_sigma_impedance
        
        # Total reconstruction loss (weighted by uncertainties)
        recon_loss = heatmap_loss + occupancy_loss + impedance_loss
        
        # KL divergence loss with numerical stability and proper scaling
        # Clamp logvar to prevent exp overflow
        logvar_clamped = torch.clamp(logvar, min=-10, max=10)
        
        # Basic KL divergence (per latent dimension, averaged over batch)
        kl_per_dim = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
        
        # Scale KL loss to be proportional to reconstruction space dimensionality
        # This ensures KL and reconstruction losses are at comparable scales
        # Total reconstruction elements per sample:
        # Heatmap: 64*64*2 = 8,192, Occupancy: 7*8*1 = 56, Impedance: 231
        total_recon_elements = 64 * 64 * 2 + 7 * 8 * 1 + 231  # 8,479 elements
        latent_elements = mu.size(1)  # latent_dim
        
        # Scale KL to match reconstruction scale: larger reconstruction space → larger KL penalty
        kl_scaling_factor = total_recon_elements / latent_elements
        kl_loss = kl_per_dim * kl_scaling_factor
        
        # Total loss with annealed KL weight  
        effective_kl_weight = self.kl_weight * kl_weight_multiplier
        total_loss = recon_loss + effective_kl_weight * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'heatmap_loss': heatmap_loss,
            'occupancy_loss': occupancy_loss,
            'impedance_loss': impedance_loss,
            'kl_loss': kl_loss,
            # Additional debugging info for uncertainty-based weighting
            'heatmap_loss_raw': heatmap_loss_raw,
            'occupancy_loss_raw': occupancy_loss_raw,
            'impedance_loss_raw': impedance_loss_raw,
            'sigma_heatmap': torch.sqrt(sigma2_heatmap),
            'sigma_occupancy': torch.sqrt(sigma2_occupancy),
            'sigma_impedance': torch.sqrt(sigma2_impedance),
            'uncertainty_reg_heatmap': 0.5 * self.log_sigma_heatmap,
            'uncertainty_reg_occupancy': 0.5 * self.log_sigma_occupancy,
            'uncertainty_reg_impedance': 0.5 * self.log_sigma_impedance
        }
