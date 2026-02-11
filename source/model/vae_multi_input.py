"""
Variational Autoencoder (VAE) with Multi-Input, Multi-Output Architecture
featuring Product of Experts (PoE) Latent Space Fusion

Architecture Overview:
    - Three independent expert encoders (one per modality)
    - Each expert directly produces latent distribution parameters (μ, σ)
    - Product of Experts (PoE) fusion: combines expert distributions using precision-weighted averaging
    - Deep fusion bottleneck (3 FC layers): learns non-linear correlations between modalities
    - Three independent decoders reconstruct each modality from latent space

Inputs:
    1. Heatmap: 64x64x2
    2. Occupancy Map: 7x8x1 (binary)
    3. Impedance Vector: 231x1

Outputs:
    1. Heatmap: 64x64x2
    2. Occupancy Map: 7x8x1
    3. Impedance Vector: 231x1

Key Benefits:
    - Latent space equally informed by all three modalities via PoE precision weighting
    - Deep fusion bottleneck preserves non-linear correlations (1D signal ↔ 2D image relationships)
    - Prevents one modality from dominating the latent representation
    - Better captures complex cross-modal dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, cast, Optional


def _init_weights(module):
    """Shared weight initialization to prevent NaN"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
        module.eps = 1e-5
        module.momentum = 0.1
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def _check_nan_inf(tensor: torch.Tensor, name: str, input_tensor: torch.Tensor):
    """Check for NaN/Inf in tensors and raise descriptive error"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"NaN/Inf in {name}. Input stats: min={input_tensor.min()}, max={input_tensor.max()}, mean={input_tensor.mean()}")


class SelfAttention2D(nn.Module):
    """Self-Attention for 2D feature maps - captures global spatial relationships"""
    
    def __init__(self, in_channels: int):
        super(SelfAttention2D, self).__init__()
        # Reduce channels for Q and K to save computation
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # Learnable scaling parameter for residual connection
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.size()
        
        # Generate Q, K, V
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        k = self.key(x).view(B, -1, H * W)  # (B, C//8, HW)
        v = self.value(x).view(B, -1, H * W)  # (B, C, HW)
        
        # Compute attention weights: Q * K^T
        attn = torch.bmm(q, k)  # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable scaling
        return self.gamma * out + x


class SelfAttention1D(nn.Module):
    """Self-Attention for 1D sequences - captures relationships in vector features"""
    
    def __init__(self, in_features: int):
        super(SelfAttention1D, self).__init__()
        # Simple MLP-based attention for 1D features
        self.attention_fc = nn.Linear(in_features, in_features)
        # Learnable scaling parameter for residual connection
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F) where F is feature dimension
        Returns:
            out: (B, F)
        """
        # Compute attention-weighted features
        attn_weights = torch.sigmoid(self.attention_fc(x))  # (B, F)
        out = x * attn_weights  # Element-wise gating
        
        # Residual connection with learnable scaling
        return self.gamma * out + x


class CrossAttention(nn.Module):
    """Cross-Attention: allows one modality to attend to another (e.g., heatmap attends to impedance)"""
    
    def __init__(self, query_channels: int, kv_dim: int, embed_dim: int = 128):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        
        # Project 2D queries to embedding space
        self.query_proj = nn.Conv2d(query_channels, embed_dim, kernel_size=1)
        # Project 1D key/value to embedding space
        self.key_proj = nn.Linear(kv_dim, embed_dim)
        self.value_proj = nn.Linear(kv_dim, embed_dim)
        # Project attention output back to original channels
        self.out_proj = nn.Conv2d(embed_dim, query_channels, kernel_size=1)
        
        # Learnable scaling for residual
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, query_features: torch.Tensor, kv_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_features: (B, C_q, H, W) - spatial features (e.g., heatmap)
            kv_features: (B, C_kv) - vector features (e.g., impedance)
        Returns:
            out: (B, C_q, H, W) - attended features
        """
        B, C_q, H, W = query_features.size()
        
        # Project queries from spatial features
        q = self.query_proj(query_features)  # (B, embed_dim, H, W)
        q = q.view(B, self.embed_dim, H * W).permute(0, 2, 1)  # (B, HW, embed_dim)
        
        # Project key and value from 1D features
        k = self.key_proj(kv_features).unsqueeze(1)  # (B, 1, embed_dim)
        v = self.value_proj(kv_features).unsqueeze(1)  # (B, 1, embed_dim)
        
        # Compute attention: Q * K^T
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, HW, 1)
        attn = F.softmax(attn, dim=1)  # Softmax over spatial positions
        
        # Apply attention to values
        out = torch.bmm(attn, v)  # (B, HW, embed_dim)
        out = out.permute(0, 2, 1).view(B, self.embed_dim, H, W)  # (B, embed_dim, H, W)
        
        # Project back to original channels
        out = self.out_proj(out)  # (B, C_q, H, W)
        
        # Residual connection with learnable scaling
        return self.gamma * out + query_features


class HeatmapEncoder(nn.Module):
    """Expert encoder for 64x64x2 heatmap input - produces latent distribution"""
    
    def __init__(self, latent_dim: int = 128):
        super(HeatmapEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Conv layers to compress 64x64x2 to feature vector
        self.conv1 = nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1)  # 32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 16x16
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 8x8
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 4x4
        self.bn4 = nn.BatchNorm2d(256)
        
        # Self-Attention after last conv block (before flattening)
        # This helps encoder "summarize" global structure
        self.self_attn = SelfAttention2D(256)
        
        # Flatten: 256 * 4 * 4 = 4096
        self.fc_hidden = nn.Linear(256 * 4 * 4, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        self.apply(_init_weights)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, 2, 64, 64)
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 256, 4, 4)
        
        # Apply self-attention to capture global structure before flattening
        x = self.self_attn(x)  # (B, 256, 4, 4)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_hidden(x))
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class OccupancyEncoder(nn.Module):
    """Expert encoder for 7x8x1 binary occupancy map - produces latent distribution"""
    
    def __init__(self, latent_dim: int = 128):
        super(OccupancyEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Conv layers for small spatial input
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 7x8
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 7x8
        self.bn2 = nn.BatchNorm2d(32)
        
        # Self-Attention for occupancy spatial features
        self.self_attn = SelfAttention2D(32)
        
        self.fc_hidden = nn.Linear(32 * 7 * 8, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.apply(_init_weights)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, 1, 7, 8)
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 32, 7, 8)
        
        # Apply self-attention to capture spatial relationships
        x = self.self_attn(x)  # (B, 32, 7, 8)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_hidden(x))
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ImpedanceEncoder(nn.Module):
    """Expert encoder for 231x1 impedance vector - produces latent distribution"""
    
    def __init__(self, latent_dim: int = 128):
        super(ImpedanceEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # MLP for 1D vector
        self.fc1 = nn.Linear(231, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Self-Attention for impedance frequency features
        self.self_attn = SelfAttention1D(128)
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        self.apply(_init_weights)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, 231)
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))  # (B, 128)
        
        # Apply self-attention to capture frequency relationships
        x = self.self_attn(x)  # (B, 128)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class PoEFusion(nn.Module):
    """
    Product of Experts (PoE) Fusion Layer
    
    Combines multiple Gaussian distributions from different modality experts
    using the product of Gaussians formula. This ensures the latent space is 
    equally informed by all modalities while preserving non-linear correlations.
    """
    
    def __init__(self, latent_dim: int = 128, num_experts: int = 3):
        super(PoEFusion, self).__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        
        # Prior variance for numerical stability
        self.register_buffer('prior_var', torch.tensor([0.1]))
        
    def forward(self, expert_mus: torch.Tensor, expert_logvars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse expert distributions using Product of Experts
        
        Args:
            expert_mus: (num_experts, batch_size, latent_dim)
            expert_logvars: (num_experts, batch_size, latent_dim)
        
        Returns:
            fused_mu: (batch_size, latent_dim)
            fused_logvar: (batch_size, latent_dim)
        """
        # Convert logvar to variance with clamping for numerical stability
        expert_logvars_clamped = torch.clamp(expert_logvars, min=-10, max=10)
        expert_vars = torch.exp(expert_logvars_clamped)  # (num_experts, batch_size, latent_dim)
        
        # Compute precision (inverse variance): τ = 1/σ²
        expert_precs = 1.0 / (expert_vars + 1e-8)  # (num_experts, batch_size, latent_dim)
        
        
        # Prior precision for regularization
        prior_var_tensor = cast(torch.Tensor, self.prior_var)
        prior_prec = 1.0 / (prior_var_tensor + 1e-8)
        
        # Product of experts: combine precisions
        # τ_combined = τ_prior + Σ(τ_i - τ_prior)
        combined_prec = prior_prec + torch.sum(expert_precs - prior_prec, dim=0)  # (batch_size, latent_dim)
        combined_prec = torch.clamp(combined_prec, min=1e-8)  # Ensure positive precision
        
        # Fused variance
        fused_var = 1.0 / combined_prec  # (batch_size, latent_dim)
        fused_logvar = torch.clamp(torch.log(fused_var + 1e-10), min=-10, max=10)
        
        # Fused mean: precision-weighted average
        # μ_combined = (τ_prior * μ_prior + Σ(τ_i * μ_i)) / τ_combined
        weighted_mu = torch.sum(expert_precs * expert_mus, dim=0)  # (batch_size, latent_dim)
        fused_mu = fused_var * weighted_mu  # (batch_size, latent_dim)
        
        return fused_mu, fused_logvar


class Encoder(nn.Module):
    """
    Unified encoder with Product of Experts (PoE) latent space fusion
    
    1. Each modality expert independently produces μ and σ
    2. Expert distributions are combined using PoE (precision-weighted fusion)
    3. Deep fusion bottleneck (2-3 dense layers) learns non-linear correlations
    4. Final latent distribution parameters generated from fused representation
    """
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 512):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Individual expert encoders for each modality
        self.heatmap_enc = HeatmapEncoder(latent_dim)
        self.occupancy_enc = OccupancyEncoder(latent_dim)
        self.impedance_enc = ImpedanceEncoder(latent_dim)
        
        # Product of Experts fusion
        self.poe_fusion = PoEFusion(latent_dim, num_experts=3)
        
        # Deep fusion bottleneck (2-3 dense layers for non-linear correlation learning)
        # Takes fused latent params (μ and σ) as input
        self.fusion_fc1 = nn.Linear(latent_dim * 2, hidden_dim)  # μ and logvar concatenated
        self.fusion_bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fusion_bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.fusion_fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.fusion_bn3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Final latent space distribution
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Initialize fusion layers
        self.apply(_init_weights)
        
    def forward(self, heatmap: torch.Tensor, occupancy: torch.Tensor, 
                impedance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            heatmap: (batch_size, 2, 64, 64)
            occupancy: (batch_size, 1, 7, 8)
            impedance: (batch_size, 231)
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        # Encode each modality to get expert distributions
        h_mu, h_logvar = self.heatmap_enc(heatmap)
        o_mu, o_logvar = self.occupancy_enc(occupancy)
        i_mu, i_logvar = self.impedance_enc(impedance)
        
        # Check for NaN/Inf in encoder outputs
        _check_nan_inf(h_mu, "heatmap encoder mu", heatmap)
        _check_nan_inf(o_mu, "occupancy encoder mu", occupancy)
        _check_nan_inf(i_mu, "impedance encoder mu", impedance)
        
        # Stack expert distributions for PoE fusion
        expert_mus = torch.stack([h_mu, o_mu, i_mu], dim=0)          # (3, batch_size, latent_dim)
        expert_logvars = torch.stack([h_logvar, o_logvar, i_logvar], dim=0)  # (3, batch_size, latent_dim)
        
        # Product of Experts: fuse expert distributions
        fused_mu, fused_logvar = self.poe_fusion(expert_mus, expert_logvars)  # (batch_size, latent_dim)
        
        # Deep fusion bottleneck: learn non-linear correlations between modalities
        # Concatenate fused μ and logvar
        fused_combined = torch.cat([fused_mu, fused_logvar], dim=1)  # (batch_size, latent_dim * 2)
        
        # Pass through 3-layer fusion bottleneck
        x = F.relu(self.fusion_bn1(self.fusion_fc1(fused_combined)))
        x = F.relu(self.fusion_bn2(self.fusion_fc2(x)))
        x = F.relu(self.fusion_bn3(self.fusion_fc3(x)))
        
        # Generate final latent distribution
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent space"""
        # Clamp logvar for numerical stability
        logvar_clamped = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


class HeatmapDecoder(nn.Module):
    """Decoder for 64x64x2 heatmap output with self-attention and cross-attention"""
    
    def __init__(self, latent_dim: int = 128, use_cross_attn: bool = True):
        super(HeatmapDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.use_cross_attn = use_cross_attn
        
        # FC layer to expand latent to 4x4x256
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.bn_fc = nn.BatchNorm1d(256 * 4 * 4)
        
        # Upsample + conv blocks to avoid checkerboard artifacts
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # Self-Attention at 16x16 resolution - sets "global blueprint"
        self.self_attn = SelfAttention2D(128)
        
        # Cross-Attention: heatmap attends to impedance features
        if self.use_cross_attn:
            self.cross_attn = CrossAttention(query_channels=128, kv_dim=latent_dim, embed_dim=128)
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        
    def forward(self, z: torch.Tensor, impedance_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z: (batch_size, latent_dim)
            impedance_features: (batch_size, latent_dim) - optional for cross-attention
        Returns:
            heatmap: (batch_size, 2, 64, 64)
        """
        x = F.relu(self.bn_fc(self.fc(z)))
        x = x.view(x.size(0), 256, 4, 4)  # 4x4
        
        x = self.upsample(x)  # 8x8
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 128, 8, 8)
        
        x = self.upsample(x)  # 16x16 - optimal resolution for attention
        
        # Apply self-attention to set global blueprint
        x = self.self_attn(x)  # (B, 128, 16, 16)
        
        # Apply cross-attention: let heatmap attend to impedance features
        if self.use_cross_attn and impedance_features is not None:
            x = self.cross_attn(x, impedance_features)  # (B, 128, 16, 16)
        
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, 16, 16)
        x = self.upsample(x)  # 32x32
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 32, 32, 32)
        x = self.upsample(x)  # 64x64
        x = torch.tanh(self.conv4(x))  # (B, 2, 64, 64)
        
        return x


class OccupancyDecoder(nn.Module):
    """Decoder for 7x8x1 binary occupancy map with self-attention and cross-attention"""
    
    def __init__(self, latent_dim: int = 128, use_cross_attn: bool = True):
        super(OccupancyDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.use_cross_attn = use_cross_attn
        
        # FC layer
        self.fc = nn.Linear(latent_dim, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 32 * 7 * 8)
        self.bn_fc2 = nn.BatchNorm1d(32 * 7 * 8)
        
        # Self-Attention at 7x8x32 resolution
        self.self_attn = SelfAttention2D(32)
        
        # Cross-Attention: occupancy attends to impedance features
        if self.use_cross_attn:
            self.cross_attn = CrossAttention(query_channels=32, kv_dim=latent_dim, embed_dim=64)
        
        # Conv layers for refinement
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, z: torch.Tensor, impedance_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z: (batch_size, latent_dim)
            impedance_features: (batch_size, latent_dim) - optional for cross-attention
        Returns:
            occupancy: (batch_size, 1, 7, 8)
        """
        x = F.relu(self.bn_fc(self.fc(z)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = x.view(x.size(0), 32, 7, 8)  # (B, 32, 7, 8)
        
        # Apply self-attention
        x = self.self_attn(x)  # (B, 32, 7, 8)
        
        # Apply cross-attention: let occupancy attend to impedance features
        if self.use_cross_attn and impedance_features is not None:
            x = self.cross_attn(x, impedance_features)  # (B, 32, 7, 8)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)  # Raw logits; apply sigmoid outside when needed
        
        return x


class ImpedanceDecoder(nn.Module):
    """Decoder for 231x1 impedance vector"""
    
    def __init__(self, latent_dim: int = 128):
        super(ImpedanceDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(latent_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 231)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch_size, latent_dim)
        Returns:
            impedance: (batch_size, 231)
        """
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    """Unified decoder with fully symmetric cross-modal attention (all modalities inform each other)"""
    
    def __init__(self, latent_dim: int = 128, use_cross_attn: bool = True):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.use_cross_attn = use_cross_attn
        
        self.heatmap_dec = HeatmapDecoder(latent_dim, use_cross_attn=False)  # Will add multi-modal attention separately
        self.occupancy_dec = OccupancyDecoder(latent_dim, use_cross_attn=False)
        self.impedance_dec = ImpedanceDecoder(latent_dim)
        
        # Multi-modal cross-attention: each modality attends to the other two
        if use_cross_attn:
            # For Heatmap: attend to Impedance + Occupancy
            self.heatmap_imp_attn = CrossAttention(query_channels=128, kv_dim=latent_dim, embed_dim=128)
            self.heatmap_occ_attn = CrossAttention(query_channels=128, kv_dim=latent_dim, embed_dim=128)
            
            # For Occupancy: attend to Impedance + Heatmap
            self.occ_imp_attn = CrossAttention(query_channels=32, kv_dim=latent_dim, embed_dim=64)
            self.occ_heat_attn = CrossAttention(query_channels=32, kv_dim=latent_dim, embed_dim=64)
            
            # Feature projection layers
            self.impedance_proj = nn.Linear(231, latent_dim)
            self.heatmap_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Global pool 128x16x16 → 128x1x1
                nn.Flatten(),  # → 128
                nn.Linear(128, latent_dim)
            )
            self.occupancy_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Global pool 32x7x8 → 32x1x1
                nn.Flatten(),  # → 32
                nn.Linear(32, latent_dim)
            )
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sequential cross-attention decoding to avoid circular dependencies:
        Stage 1: Decode all modalities to intermediate representations
        Stage 2: Apply sequential cross-attention (avoids using stale features)
        Stage 3: Final refinement layers
        
        Sequential order prevents circular gradient dependencies:
        1. Heatmap attends to Impedance
        2. Heatmap attends to Occupancy  
        3. Occupancy attends to Impedance
        4. Occupancy attends to updated Heatmap (not stale features)
        
        Args:
            z: (batch_size, latent_dim)
        Returns:
            heatmap: (batch_size, 2, 64, 64)
            occupancy_logits: (batch_size, 1, 7, 8)
            impedance: (batch_size, 231)
        """
        # Stage 1: Decode to intermediate features (before final conv layers)
        # Impedance: fully decoded (1D, no spatial refinement needed)
        impedance = self.impedance_dec(z)
        
        # Heatmap: decode to 16x16x128 (before final upsampling)
        h = F.relu(self.heatmap_dec.bn_fc(self.heatmap_dec.fc(z)))
        h = h.view(h.size(0), 256, 4, 4)
        h = self.heatmap_dec.upsample(h)
        h = F.relu(self.heatmap_dec.bn1(self.heatmap_dec.conv1(h)))
        h = self.heatmap_dec.upsample(h)  # 16x16x128
        h = self.heatmap_dec.self_attn(h)  # Self-attention
        
        # Occupancy: decode to 7x8x32 (before final conv layers)
        o = F.relu(self.occupancy_dec.bn_fc(self.occupancy_dec.fc(z)))
        o = F.relu(self.occupancy_dec.bn_fc2(self.occupancy_dec.fc2(z)))
        o = o.view(o.size(0), 32, 7, 8)
        o = self.occupancy_dec.self_attn(o)  # Self-attention
        
        # Stage 2: Sequential cross-attention (avoids circular dependencies)
        if self.use_cross_attn:
            # Step 1: Project impedance features (constant, used by both)
            imp_feat = self.impedance_proj(impedance)  # (B, latent_dim)
            
            # Step 2: Project initial occupancy features (before heatmap gets updated)
            occ_feat_initial = self.occupancy_proj(o)  # (B, latent_dim)
            
            # Step 3: Heatmap attends to Impedance + initial Occupancy
            h = self.heatmap_imp_attn(h, imp_feat)
            h = self.heatmap_occ_attn(h, occ_feat_initial)
            
            # Step 4: Project UPDATED heatmap features (after attention)
            heat_feat_updated = self.heatmap_proj(h)  # (B, latent_dim)
            
            # Step 5: Occupancy attends to Impedance + UPDATED Heatmap
            o = self.occ_imp_attn(o, imp_feat)
            o = self.occ_heat_attn(o, heat_feat_updated)  # Uses fresh heatmap features!
        
        # Stage 3: Final refinement layers
        # Heatmap: upsample to 64x64
        h = F.relu(self.heatmap_dec.bn2(self.heatmap_dec.conv2(h)))
        h = self.heatmap_dec.upsample(h)
        h = F.relu(self.heatmap_dec.bn3(self.heatmap_dec.conv3(h)))
        h = self.heatmap_dec.upsample(h)
        heatmap = torch.tanh(self.heatmap_dec.conv4(h))
        
        # Occupancy: final conv layers
        o = F.relu(self.occupancy_dec.bn1(self.occupancy_dec.conv1(o)))
        occupancy = self.occupancy_dec.conv2(o)
        
        return heatmap, occupancy, impedance


class MultiInputVAE(nn.Module):
    """
    Variational Autoencoder with three inputs and three outputs
    
    Inputs:
        - heatmap: (batch_size, 2, 64, 64)
        - occupancy: (batch_size, 1, 7, 8)
        - impedance: (batch_size, 231)
    
    Outputs:
        - reconstructed_heatmap: (batch_size, 2, 64, 64)
        - reconstructed_occupancy: (batch_size, 1, 7, 8)
        - reconstructed_impedance: (batch_size, 231)
    """
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 512, use_cross_attn: bool = True):
        super(MultiInputVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = Encoder(latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, use_cross_attn=use_cross_attn)
        
    def encode(self, heatmap: torch.Tensor, occupancy: torch.Tensor,
               impedance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs to latent space"""
        mu, logvar = self.encoder(heatmap, occupancy, impedance)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode from latent space"""
        heatmap, occupancy, impedance = self.decoder(z)
        return heatmap, occupancy, impedance
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        return self.encoder.reparameterize(mu, logvar)
    
    def forward(self, heatmap: torch.Tensor, occupancy: torch.Tensor,
                impedance: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE
        
        Args:
            heatmap: (batch_size, 2, 64, 64)
            occupancy: (batch_size, 1, 7, 8)
            impedance: (batch_size, 231)
        
        Returns:
            Dictionary containing:
                - mu: latent mean
                - logvar: latent log variance
                - z: sampled latent vector
                - heatmap_recon: reconstructed heatmap
                - occupancy_recon: reconstructed occupancy
                - impedance_recon: reconstructed impedance
        """
        # Encode
        mu, logvar = self.encoder(heatmap, occupancy, impedance)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode
        heatmap_recon, occupancy_logits, impedance_recon = self.decoder(z)
        occupancy_recon = torch.sigmoid(occupancy_logits)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'heatmap_recon': heatmap_recon,
            'occupancy_recon': occupancy_recon,
            'occupancy_logits': occupancy_logits,
            'impedance_recon': impedance_recon
        }

