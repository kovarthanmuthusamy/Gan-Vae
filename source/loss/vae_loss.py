"""Loss function for Multi-Input VAE"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice Loss for binary segmentation"""
    pred_probs = torch.sigmoid(pred_logits)
    pred_flat = pred_probs.view(pred_probs.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return 1.0 - dice_coeff.mean()


class VAELoss(nn.Module):
    """VAE loss with uncertainty-based weighting and KL divergence"""
    
    def __init__(self, 
                 init_log_sigma_heatmap: float = -0.5,
                 init_log_sigma_occupancy: float = -0.5, 
                 init_log_sigma_impedance: float = -0.5,
                 kl_weight: float = 0.001,
                 occupancy_loss_type: str = 'dice_bce',
                 static_occupancy_weight: Optional[float] = None,
                 static_weight_epochs: int = 50,
                 heatmap_gradient_weight: float = 0.1,  # Weight for spatial gradient loss
                 impedance_gradient_weight: float = 0.1,  # Weight for gradient loss to capture peaks
                 **kwargs):  # Unused legacy parameters
        super().__init__()
        
        self.log_sigma_heatmap = nn.Parameter(torch.tensor(init_log_sigma_heatmap, dtype=torch.float32))
        self.log_sigma_max_impedance = nn.Parameter(torch.tensor(init_log_sigma_heatmap, dtype=torch.float32))
        self.log_sigma_occupancy = nn.Parameter(torch.tensor(init_log_sigma_occupancy, dtype=torch.float32))
        self.log_sigma_impedance = nn.Parameter(torch.tensor(init_log_sigma_impedance, dtype=torch.float32))
        
        self.kl_weight = kl_weight
        self.occupancy_loss_type = occupancy_loss_type
        self.static_occupancy_weight = static_occupancy_weight
        self.static_weight_epochs = static_weight_epochs
        self.current_epoch = 0
        self.heatmap_gradient_weight = heatmap_gradient_weight
        self.impedance_gradient_weight = impedance_gradient_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        
    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch
        
    def compute_occupancy_loss(self, occupancy_logits: torch.Tensor, occupancy_target: torch.Tensor) -> torch.Tensor:
        if self.occupancy_loss_type == 'bce':
            return self.bce_loss(occupancy_logits, occupancy_target)
        elif self.occupancy_loss_type == 'dice':
            return dice_loss(occupancy_logits, occupancy_target)
        elif self.occupancy_loss_type == 'dice_bce':
            dice = dice_loss(occupancy_logits, occupancy_target)
            bce = F.binary_cross_entropy_with_logits(occupancy_logits, occupancy_target, reduction='mean')
            return 0.5 * dice + 0.5 * bce
        else:
            raise ValueError(f"Unknown occupancy_loss_type: {self.occupancy_loss_type}")
    
    def compute_spatial_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradient loss for 2D heatmaps to capture edges and fine details
        
        Args:
            pred: Predicted heatmap [batch_size, channels, height, width]
            target: Target heatmap [batch_size, channels, height, width]
        
        Returns:
            Spatial gradient loss (scalar)
        """
        # Compute gradients in x direction (horizontal)
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # Compute gradients in y direction (vertical)
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # L1 loss on spatial gradients
        grad_loss_x = self.mae_loss(pred_grad_x, target_grad_x)
        grad_loss_y = self.mae_loss(pred_grad_y, target_grad_y)
        
        # Average both directions
        return 0.5 * (grad_loss_x + grad_loss_y)
    
    def compute_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss on gradients to capture peaks and local variations
        
        Args:
            pred: Predicted impedance [batch_size, 231]
            target: Target impedance [batch_size, 231]
        
        Returns:
            Gradient loss (scalar)
        """
        # Compute gradients (differences between adjacent points)
        pred_grad = pred[:, 1:] - pred[:, :-1]
        target_grad = target[:, 1:] - target[:, :-1]
        
        # L1 loss on gradients
        grad_loss_l1 = self.mae_loss(pred_grad, target_grad)
        
        # MSE loss on gradients (penalizes large errors more, important for peaks)
        grad_loss_mse = self.mse_loss(pred_grad, target_grad)
        
        # Combine both (L1 for robustness, MSE for emphasizing peaks)
        return 0.5 * grad_loss_l1 + 0.5 * grad_loss_mse

    def forward(self, outputs: Dict[str, torch.Tensor],
                heatmap_target_norm: torch.Tensor,
                max_impedance_target_std: torch.Tensor,
                occupancy_target: torch.Tensor,
                impedance_target: torch.Tensor,
                max_impedance_mean: float,
                max_impedance_std: float,
                kl_weight_multiplier: float = 1.0,
                use_physical_loss: bool = True) -> Dict[str, torch.Tensor]:
        
        mu = outputs['mu']
        logvar = outputs['logvar']
        heatmap_norm_recon = outputs['heatmap_recon']
        max_impedance_std_recon = outputs['max_impedance_recon']
        occupancy_logits = outputs.get('occupancy_logits', outputs['occupancy_recon'])
        impedance_recon = outputs['impedance_recon']
        
        # Compute losses
        if use_physical_loss:
            max_impedance_recon = max_impedance_std_recon * max_impedance_std + max_impedance_mean
            max_impedance_target = max_impedance_target_std * max_impedance_std + max_impedance_mean
            
            heatmap_physical_pred = heatmap_norm_recon * max_impedance_recon.view(-1, 1, 1, 1)
            heatmap_physical_target = heatmap_target_norm * max_impedance_target.view(-1, 1, 1, 1)
            
            # Heatmap loss with spatial gradient to capture fine details
            heatmap_mae = self.mae_loss(heatmap_physical_pred, heatmap_physical_target)
            heatmap_spatial_grad = self.compute_spatial_gradient_loss(heatmap_physical_pred, heatmap_physical_target)
            heatmap_physical_loss_raw = heatmap_mae + self.heatmap_gradient_weight * heatmap_spatial_grad
            
            max_impedance_loss_raw = self.mae_loss(max_impedance_recon, max_impedance_target)
            heatmap_loss_raw = heatmap_physical_loss_raw + 0.5 * max_impedance_loss_raw
        else:
            # Heatmap loss with spatial gradient
            heatmap_mae = self.mae_loss(heatmap_norm_recon, heatmap_target_norm)
            heatmap_spatial_grad = self.compute_spatial_gradient_loss(heatmap_norm_recon, heatmap_target_norm)
            heatmap_loss_raw = heatmap_mae + self.heatmap_gradient_weight * heatmap_spatial_grad
            
            max_impedance_loss_raw = self.mae_loss(max_impedance_std_recon, max_impedance_target_std)
        
        occupancy_loss_raw = self.compute_occupancy_loss(occupancy_logits, occupancy_target)
        
        # Impedance loss with gradient component to capture peaks
        impedance_mae = self.mae_loss(impedance_recon, impedance_target)
        impedance_grad_loss = self.compute_gradient_loss(impedance_recon, impedance_target)
        impedance_loss_raw = impedance_mae + self.impedance_gradient_weight * impedance_grad_loss
        
        # Apply uncertainty-based weighting
        log_sigma_hm_clamped = torch.clamp(self.log_sigma_heatmap, min=-3.0, max=2.0)
        log_sigma_occ_clamped = torch.clamp(self.log_sigma_occupancy, min=-3.0, max=2.0)
        log_sigma_imp_clamped = torch.clamp(self.log_sigma_impedance, min=-3.0, max=2.0)
        
        sigma2_heatmap = torch.exp(log_sigma_hm_clamped)
        sigma2_occupancy = torch.exp(log_sigma_occ_clamped)
        sigma2_impedance = torch.exp(log_sigma_imp_clamped)
        
        heatmap_loss = heatmap_loss_raw / (2.0 * sigma2_heatmap)
        
        if self.static_occupancy_weight is not None and self.current_epoch < self.static_weight_epochs:
            occupancy_loss = self.static_occupancy_weight * occupancy_loss_raw
        else:
            occupancy_loss = occupancy_loss_raw / (2.0 * sigma2_occupancy)
        
        impedance_loss = impedance_loss_raw / (2.0 * sigma2_impedance)
        recon_loss = heatmap_loss + occupancy_loss + impedance_loss
        
        # KL divergence
        logvar_clamped = torch.clamp(logvar, min=-10, max=10)
        kl_per_dim = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
        
        total_recon_elements = 64 * 64 * 2 + 7 * 8 * 1 + 231
        kl_scaling_factor = total_recon_elements / mu.size(1)
        kl_loss = kl_per_dim * kl_scaling_factor
        
        total_loss = recon_loss + self.kl_weight * kl_weight_multiplier * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'heatmap_loss': heatmap_loss,
            'max_impedance_loss': max_impedance_loss_raw if use_physical_loss else torch.tensor(0.0),
            'occupancy_loss': occupancy_loss,
            'impedance_loss': impedance_loss,
            'kl_loss': kl_loss,
            'heatmap_loss_raw': heatmap_loss_raw,
            'heatmap_spatial_grad_loss': heatmap_spatial_grad,
            'max_impedance_loss_raw': max_impedance_loss_raw,
            'occupancy_loss_raw': occupancy_loss_raw,
            'impedance_loss_raw': impedance_loss_raw,
            'impedance_grad_loss': impedance_grad_loss,
            'sigma_heatmap': torch.sqrt(sigma2_heatmap),
            'sigma_occupancy': torch.sqrt(sigma2_occupancy),
            'sigma_impedance': torch.sqrt(sigma2_impedance)
        }
