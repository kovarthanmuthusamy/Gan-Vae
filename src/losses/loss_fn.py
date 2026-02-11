import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from torch.autograd import grad

def gradient_penalty(critic, real_heatmap, real_occupancy, real_impedance,
                     fake_heatmap, fake_occupancy, fake_impedance,
                     mask=None,
                     device="cuda"):

    B = real_heatmap.size(0)

    # Random interpolation coefficient (shared across modalities for stability)
    alpha = torch.rand(B, 1, 1, 1, device=device)

    # Interpolated samples
    hat_heatmap = alpha * real_heatmap + (1 - alpha) * fake_heatmap
    hat_occupancy = alpha * real_occupancy + (1 - alpha) * fake_occupancy
    hat_impedance = alpha.view(B, 1) * real_impedance + (1 - alpha.view(B, 1)) * fake_impedance
    
    if mask is not None:
        hat_heatmap = hat_heatmap * mask

    hat_heatmap.requires_grad_(True)
    hat_occupancy.requires_grad_(True)
    hat_impedance.requires_grad_(True)

    # Critic output
    d_hat = critic(hat_heatmap, hat_occupancy, hat_impedance)

    # Gradient w.r.t inputs
    gradients = grad(
        outputs=d_hat,
        inputs=[hat_heatmap, hat_occupancy, hat_impedance],
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )

    # Flatten gradients
    grad_h = gradients[0].view(B, -1)
    grad_o = gradients[1].view(B, -1)
    grad_i = gradients[2].view(B, -1)

    grad_all = torch.cat([grad_h, grad_o, grad_i], dim=1)

    # L2 norm
    grad_norm = grad_all.norm(2, dim=1)
    grad_norm_mean = grad_norm.mean().item()

    # Gradient penalty
    gp = ((grad_norm - 1) ** 2).mean()

    return gp, grad_norm_mean

def critic_loss(critic,
                real_heatmap, real_occupancy, real_impedance,
                fake_heatmap, fake_occupancy, fake_impedance,
                lambda_gp=10.0,
                device="cuda",
                mask=None,
                epsilon_drift: float = 1e-3):

    real_score = critic(real_heatmap, real_occupancy, real_impedance)
    fake_score = critic(fake_heatmap.detach(), fake_occupancy.detach(), fake_impedance.detach())

    gp, grad_norm_mean = gradient_penalty(
        critic,
        real_heatmap, real_occupancy, real_impedance,
        fake_heatmap, fake_occupancy, fake_impedance,
        mask=mask,
        device=device
    )

    # Drift regularization to prevent critic output explosion
    drift = epsilon_drift * (real_score.pow(2).mean())

    loss_D = fake_score.mean() - real_score.mean() + lambda_gp * gp + drift

    # Return extra diagnostics: gp and score means
    return loss_D, grad_norm_mean, gp.item(), real_score.mean().item(), fake_score.mean().item()


def generator_loss(critic, fake_heatmap, fake_occupancy, fake_impedance):
    loss_G = -critic(fake_heatmap, fake_occupancy, fake_impedance).mean()
    return loss_G

def binarize_occupancy(occupancy, threshold=0.5):
    """
    Binarize occupancy output during inference.
    Args:
        occupancy: (batch, 1, 7, 8) - continuous values from sigmoid [0, 1]
        threshold: Threshold for binarization (default: 0.5)
    Returns:
        Binarized occupancy: (batch, 1, 7, 8) - binary values [0, 1]
    """
    return (occupancy > threshold).float()

def binarize_impedance(impedance, threshold=0.5):
    """
    Binarize impedance output during inference.
    Args:
        impedance: (batch, 231) - continuous values from sigmoid [0, 1]
        threshold: Threshold for binarization (default: 0.5)
    Returns:
        Binarized impedance: (batch, 231) - binary values [0, 1]
    """
    return (impedance > threshold).float()


def feature_matching_loss(critic,
                          real_heatmap, real_occupancy, real_impedance,
                          fake_heatmap, fake_occupancy, fake_impedance):
    """
    Feature matching loss using intermediate features from the critic's fusion_conv2 layer.
    Computes MAE (Mean Absolute Error) between feature maps of real and fake samples.
    
    This acts as a reconstruction loss in feature space rather than pixel space,
    encouraging the generator to produce samples that have similar intermediate
    representations to real data.
    
    """
    # Get intermediate features from critic for real samples
    with torch.no_grad():
        _, real_intermediates = critic.forward_with_intermediates(
            real_heatmap, real_occupancy, real_impedance
        )
    
    # Get intermediate features from critic for fake samples (with gradients for generator training)
    _, fake_intermediates = critic.forward_with_intermediates(
        fake_heatmap, fake_occupancy, fake_impedance
    )
    
    # Extract features after fusion_conv2 (shape: B, 32, 16, 16)
    real_features = real_intermediates['after_conv2']
    fake_features = fake_intermediates['after_conv2']
    
    # Compute MAE (Mean Absolute Error) between feature maps
    loss = torch.nn.functional.l1_loss(fake_features, real_features.detach())
    
    return loss
