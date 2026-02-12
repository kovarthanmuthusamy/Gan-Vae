"""
Variational Autoencoder (VAE) with multi-input, multi-output architecture
featuring mid-layer fusion at 8x8 resolution.

Architecture Overview:
    ENCODER (Mid-Layer Fusion):
    - Three independent branches process to 8x8x128:
      * Heatmap: 64x64x2 → Conv layers → 8x8x128
      * Occupancy: 7x8x1 → Conv + Upsample → 8x8x128
      * Impedance: 231 → MLP (231→512→8192) → reshape → 8x8x128
    - Feature concatenation at 8x8: 384 channels (3 * 128)
    - Fusion conv: 384 → 128 channels (1x1 conv)
    - Self-attention at 8x8: captures cross-modal spatial relationships
    - Continue encoding: 8x8x128 → 4x4x256 → latent space
    
    DECODER:
    - Three independent decoders reconstruct each modality from latent space
    - Self-attention within each decoder captures global structure without
      cross-modal attention layers

Inputs:
    1. Heatmap: 64x64x2
    2. Occupancy Map: 7x8x1 (binary)
    3. Impedance Vector: 231x1

Outputs:
    1. Heatmap: 64x64x2
    2. Occupancy Map: 7x8x1
    3. Impedance Vector: 231x1

Key Benefits:
    - 38.7% fewer parameters than early fusion (7.4M vs 12M encoder params)
    - Each modality processes with own early layers (preserves characteristics)
    - Fusion at abstract feature level (8x8) more natural for heterogeneous data
    - Impedance projection: 4.3M params (vs 9M in early fusion, 52% reduction)
    - Self-attention AFTER fusion captures cross-modal dependencies
    - Better suited for small-to-medium datasets (less overfitting risk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, cast


ATTENTION_GAMMA_INIT = 0.05


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
        self.gamma = nn.Parameter(torch.full((1,), ATTENTION_GAMMA_INIT))
        
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
        self.gamma = nn.Parameter(torch.full((1,), ATTENTION_GAMMA_INIT))
        
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




class HeatmapBranch(nn.Module):
    """Heatmap encoder branch: 64x64x2 → 8x8x128"""
    
    def __init__(self):
        super(HeatmapBranch, self).__init__()
        
        # Conv layers to compress heatmap to 8x8x128
        self.conv1 = nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1)  # 32x32
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 16x16
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 8x8
        self.bn3 = nn.BatchNorm2d(128)
        
        self.apply(_init_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 2, 64, 64)
        Returns:
            features: (batch_size, 128, 8, 8)
        """
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 32, 32, 32)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, 16, 16)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 128, 8, 8)
        return x


class OccupancyBranch(nn.Module):
    """Occupancy encoder branch: 7x8x1 → 8x8x128"""
    
    def __init__(self):
        super(OccupancyBranch, self).__init__()
        
        # Conv layers for small spatial input
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 7x8
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 7x8
        self.bn2 = nn.BatchNorm2d(32)
        
        # Bilinear upsample to 8x8 (no parameters)
        self.upsample = nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 8x8
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 8x8
        self.bn4 = nn.BatchNorm2d(128)
        
        self.apply(_init_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 1, 7, 8)
        Returns:
            features: (batch_size, 128, 8, 8)
        """
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 16, 7, 8)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 32, 7, 8)
        x = self.upsample(x)  # (B, 32, 8, 8) - bilinear interpolation
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 64, 8, 8)
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 128, 8, 8)
        return x


class ImpedanceBranch(nn.Module):
    """Impedance encoder branch: 231x1 → 8x8x128 (via MLP then reshape)"""
    
    def __init__(self):
        super(ImpedanceBranch, self).__init__()
        
        # MLP: 231 → 512 → 8192 (= 8*8*128)
        self.fc1 = nn.Linear(231, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 8 * 8 * 128)  # 8192
        self.bn2 = nn.BatchNorm1d(8 * 8 * 128)
        
        self.apply(_init_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 231)
        Returns:
            features: (batch_size, 128, 8, 8)
        """
        x = F.relu(self.bn1(self.fc1(x)))  # (B, 512)
        x = F.relu(self.bn2(self.fc2(x)))  # (B, 8192)
        
        # Reshape to spatial: (B, 8192) → (B, 128, 8, 8)
        x = x.view(x.size(0), 128, 8, 8)
        return x


class MidLayerFusionEncoder(nn.Module):
    """Mid-layer fusion encoder: fuses modalities at 8x8 resolution before self-attention"""
    
    def __init__(self, latent_dim: int = 128):
        super(MidLayerFusionEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Three separate branches process to 8x8x128
        self.heatmap_branch = HeatmapBranch()
        self.occupancy_branch = OccupancyBranch()
        self.impedance_branch = ImpedanceBranch()
        
        # Fusion: concatenate 3 * 128 = 384 channels → reduce to 128
        self.fusion_conv = nn.Conv2d(384, 128, kernel_size=1)  # 1x1 conv
        self.fusion_bn = nn.BatchNorm2d(128)
        
        # Self-Attention at mid-layer (8x8) AFTER fusion to capture cross-modal relationships
        self.self_attn = SelfAttention2D(128)
        
        # Continue encoding
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 4x4
        self.bn4 = nn.BatchNorm2d(256)
        
        # No attention at 4x4 - too small (16 positions), regular conv sufficient
        # Self-attention at 8x8 already captured global/cross-modal relationships
        
        # Fully connected layers to latent space
        self.fc_hidden = nn.Linear(256 * 4 * 4, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
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
        # Process each modality independently to 8x8x128
        h_feat = self.heatmap_branch(heatmap)      # (B, 128, 8, 8)
        o_feat = self.occupancy_branch(occupancy)  # (B, 128, 8, 8)
        i_feat = self.impedance_branch(impedance)  # (B, 128, 8, 8)
        
        # Concatenate along channel dimension: 3 * 128 = 384 channels
        fused = torch.cat([h_feat, o_feat, i_feat], dim=1)  # (B, 384, 8, 8)
        
        # Reduce channels with 1x1 conv
        x = F.relu(self.fusion_bn(self.fusion_conv(fused)))  # (B, 128, 8, 8)
        
        # Mid-layer self-attention: captures cross-modal spatial relationships
        x = self.self_attn(x)  # (B, 128, 8, 8)
        
        # Continue encoding
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 256, 4, 4)
        
        # Flatten and project to latent space (no attention at 4x4)
        x = x.view(x.size(0), -1)  # (B, 4096)
        x = F.relu(self.fc_hidden(x))  # (B, 512)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Encoder(nn.Module):
    """
    Mid-layer fusion encoder: combines modalities at 8x8 resolution
    
    Architecture:
    1. Three independent branches process each modality to 8x8x128:
       - Heatmap: 64x64x2 → Conv layers → 8x8x128
       - Occupancy: 7x8x1 → Conv + Upsample → 8x8x128  
       - Impedance: 231 → MLP → 8x8x128
    2. Concatenate at 8x8: 384 channels (3 * 128)
    3. Fusion conv: 384 → 128 channels
    4. Self-attention at 8x8: captures cross-modal relationships
    5. Continue encoding to latent space
    
    Benefits:
    - 38.7% fewer parameters than early fusion
    - Each modality preserves its own characteristics initially
    - Fusion at abstract feature level (8x8) is more natural
    - Impedance projection: 4.3M params (vs 9M in early fusion)
    """
    
    def __init__(self, latent_dim: int = 128):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Mid-layer fusion encoder
        self.fusion_encoder = MidLayerFusionEncoder(latent_dim)
        
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
        # Process through mid-layer fusion encoder
        mu, logvar = self.fusion_encoder(heatmap, occupancy, impedance)
        
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
    """Decoder for 64x64x2 heatmap output with self-attention"""
    
    def __init__(self, latent_dim: int = 128):
        super(HeatmapDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # FC layer to expand latent to 4x4x256
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.bn_fc = nn.BatchNorm1d(256 * 4 * 4)
        
        # Upsample + conv blocks to avoid checkerboard artifacts
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # Self-Attention at 16x16 resolution - sets "global blueprint"
        self.self_attn = SelfAttention2D(128)
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        
        self.apply(_init_weights)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch_size, latent_dim)
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
        
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, 16, 16)
        x = self.upsample(x)  # 32x32
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 32, 32, 32)
        x = self.upsample(x)  # 64x64
        x = torch.tanh(self.conv4(x))  # (B, 2, 64, 64)
        
        return x


class OccupancyDecoder(nn.Module):
    """Decoder for 7x8x1 binary occupancy map with self-attention"""
    
    def __init__(self, latent_dim: int = 128):
        super(OccupancyDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # FC layer
        self.fc = nn.Linear(latent_dim, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 32 * 7 * 8)
        self.bn_fc2 = nn.BatchNorm1d(32 * 7 * 8)
        
        # Self-Attention at 7x8x32 resolution
        self.self_attn = SelfAttention2D(32)
        
        # Conv layers for refinement
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        
        self.apply(_init_weights)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch_size, latent_dim)
        Returns:
            occupancy: (batch_size, 1, 7, 8)
        """
        x = F.relu(self.bn_fc(self.fc(z)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = x.view(x.size(0), 32, 7, 8)  # (B, 32, 7, 8)
        
        # Apply self-attention
        x = self.self_attn(x)  # (B, 32, 7, 8)
        
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
        
        self.apply(_init_weights)
        
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
    """Unified decoder with per-modality self-attention"""
    
    def __init__(self, latent_dim: int = 128):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.heatmap_dec = HeatmapDecoder(latent_dim)
        self.occupancy_dec = OccupancyDecoder(latent_dim)
        self.impedance_dec = ImpedanceDecoder(latent_dim)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch_size, latent_dim)
        Returns:
            heatmap: (batch_size, 2, 64, 64)
            occupancy_logits: (batch_size, 1, 7, 8)
            impedance: (batch_size, 231)
        """
        heatmap = self.heatmap_dec(z)
        occupancy = self.occupancy_dec(z)
        impedance = self.impedance_dec(z)
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
    
    def __init__(self, latent_dim: int = 128):
        super(MultiInputVAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
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

