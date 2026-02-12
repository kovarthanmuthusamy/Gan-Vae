"""
Loss function for Multi-Input Variational Autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


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
    """Combined loss function for multi-output VAE"""
    
    def __init__(self, heatmap_weight: float = 1.0, 
                 occupancy_weight: float = 1.0,
                 impedance_weight: float = 1.0,
                 kl_weight: float = 0.001,
                 occupancy_bce_weight: float = 0.5,
                 occupancy_dice_weight: float = 0.5,
                 impedance_cosine_weight: float = 0.5,
                 impedance_mse_weight: float = 0.5):
        super(VAELoss, self).__init__()
        self.heatmap_weight = heatmap_weight
        self.occupancy_weight = occupancy_weight
        self.impedance_weight = impedance_weight
        self.kl_weight = kl_weight
        self.occupancy_bce_weight = occupancy_bce_weight
        self.occupancy_dice_weight = occupancy_dice_weight
        self.impedance_cosine_weight = impedance_cosine_weight
        self.impedance_mse_weight = impedance_mse_weight
        
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mae_loss = nn.L1Loss(reduction='mean')
    def forward(self, outputs: Dict[str, torch.Tensor],
                heatmap_target: torch.Tensor,
                occupancy_target: torch.Tensor,
                impedance_target: torch.Tensor,
                kl_weight_multiplier: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Calculate VAE loss with optional KL weight multiplier for annealing
        
        Args:
            outputs: Dictionary from VAE forward pass
            heatmap_target: Target heatmap (batch_size, 2, 64, 64)
            occupancy_target: Target occupancy (batch_size, 1, 7, 8)
            impedance_target: Target impedance (batch_size, 231)
            kl_weight_multiplier: Multiplier for KL weight (default: 1.0)
                                  Used for KL annealing (0.0 to 1.0)
        
        Returns:
            Dictionary with loss components
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
        
        # Reconstruction losses
        # SSIM loss for heatmap (perceptual similarity)
        #heatmap_loss = ssim_loss(heatmap_recon, heatmap_target)
        heatmap_loss = self.mae_loss(heatmap_recon, heatmap_target) 
        
        # Hybrid occupancy loss: BCE + Dice for better segmentation
        occupancy_bce = self.bce_loss(occupancy_logits, occupancy_target)
        #occupancy_dice = dice_loss(occupancy_logits, occupancy_target)
        occupancy_loss = occupancy_bce 
        
        # Hybrid impedance loss: Cosine Similarity (pattern/shape) + MSE (magnitude)
        #impedance_cosine = cosine_similarity_loss(impedance_recon, impedance_target)
        impedance_mse = self.mae_loss(impedance_recon, impedance_target)
        impedance_loss = impedance_mse
        
        # Weighted reconstruction loss
        recon_loss = (self.heatmap_weight * heatmap_loss +
                     self.occupancy_weight * occupancy_loss +
                     self.impedance_weight * impedance_loss)
        
        # KL divergence loss with numerical stability
        # Clamp logvar to prevent exp overflow
        logvar_clamped = torch.clamp(logvar, min=-10, max=10)
        kl_loss = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
        
        # Total loss with annealed KL weight
        effective_kl_weight = self.kl_weight * kl_weight_multiplier
        total_loss = recon_loss + effective_kl_weight * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'heatmap_loss': heatmap_loss,
            'occupancy_loss': occupancy_loss,
           # 'occupancy_bce': occupancy_bce,
            #'occupancy_dice': occupancy_dice,
            'impedance_loss': impedance_loss,
            #'impedance_cosine': impedance_cosine,
            #'impedance_mse': impedance_mse,
            'kl_loss': kl_loss
        }
