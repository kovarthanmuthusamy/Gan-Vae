import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# --------------------
# Attention Blocks
# --------------------
class SelfAttention2d(nn.Module):
    """SAGAN-style non-local self-attention for 2D feature maps.
    Computes attention over spatial positions and adds a residual.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // 8)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        theta = self.theta(x).view(b, self.inter_channels, n)        # (B, C//8, N)
        phi = self.phi(x).view(b, self.inter_channels, n)            # (B, C//8, N)
        g = self.g(x).view(b, c, n)                                  # (B, C, N)

        attn = torch.bmm(theta.transpose(1, 2), phi)                 # (B, N, N)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(g, attn.transpose(1, 2))                     # (B, C, N)
        out = out.view(b, c, h, w)
        return x + self.gamma * out

# --------------------
# Generator
# --------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, shared_dim=512):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.shared_dim = shared_dim

        # Shared 2D latent image (project noise to 2D feature map)
        # Use 128 channels at 8x8 to unify branches
        self.shared_channels = 128
        self.shared_2d_fc = nn.Sequential(
            nn.Linear(latent_dim, self.shared_channels * 8 * 8),
            nn.ReLU()
        )
        # Self-attention over shared 2D latent
        self.shared_attn = SelfAttention2d(self.shared_channels)

        # Heatmap branch (2x64x64) - impedance and mask
        self.heatmap_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1),    # 64x64x2
        )
        self.heatmap_sigmoid = nn.Sigmoid()

        # Occupancy map branch (1x7x8) derived from shared 2D latent
        self.occupancy_head = nn.Sequential(
            nn.Conv2d(self.shared_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(7, 8), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.occupancy_sigmoid = nn.Sigmoid()

        # Impedance branch (231,) from pooled shared 2D latent
        self.impedance_pool = nn.AdaptiveAvgPool2d(1)
        self.impedance_fc = nn.Sequential(
            nn.Linear(self.shared_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 231),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()

    def forward(self, z):
        # Project to shared 2D latent image
        shared_2d = self.shared_2d_fc(z)
        shared_2d = shared_2d.view(-1, self.shared_channels, 8, 8)

        # Heatmap branch (64x64x2)
        h = self.heatmap_deconv(self.shared_attn(shared_2d))
        heatmap = self.heatmap_sigmoid(h)

        # Occupancy map branch (7x8x1) from shared 2D latent
        occ = self.occupancy_head(shared_2d)
        occupancy = self.occupancy_sigmoid(occ)

        # Impedance branch (231,) via global pooling of shared 2D latent
        i = self.impedance_pool(shared_2d).view(-1, self.shared_channels)
        impedance = self.impedance_fc(i)

        return heatmap, occupancy, impedance
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for Conv/Linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# --------------------
# Critic / Discriminator
# --------------------
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # --- Fused 2D path ---
        # Upsample occupancy to 64x64
        self.occupancy_upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)

        # Project impedance vector -> 2D feature map (C_imp x 8 x 8), then upsample to 64x64
        self.imp_channels = 4
        self.impedance_proj = nn.Sequential(
            spectral_norm(nn.Linear(231, self.imp_channels * 8 * 8)),
            nn.LeakyReLU(0.2)
        )
        self.impedance_upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)

        # Convolutional fusion over concatenated channels: [heatmap(2) + occupancy(1) + imp(self.imp_channels)]
        in_ch = 2 + 1 + self.imp_channels
        # Convolutional stack with an attention block at 16x16
        self.fusion_conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, 16, 4, stride=2, padding=1)),   # 64->32
            nn.LeakyReLU(0.2),
        )
        self.fusion_conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(16, 32, 4, stride=2, padding=1)),      # 32->16
            nn.LeakyReLU(0.2),
        )
        self.attn_critic = SelfAttention2d(32)              # at 16x16
        # Apply spectral norm to attention only in Critic
        self.attn_critic.theta = spectral_norm(self.attn_critic.theta)
        self.attn_critic.phi = spectral_norm(self.attn_critic.phi)
        self.attn_critic.g = spectral_norm(self.attn_critic.g)
        
        self.fusion_conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),     # 16->8
            nn.LeakyReLU(0.2),
        )
        self.flatten = nn.Flatten()

        # Final classifier
        self.joint_fc = nn.Sequential(
            spectral_norm(nn.Linear(64*8*8, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 1))
        )
        
        # Initialize weights
        self._init_weights()

    def forward(self, heatmap, occupancy, impedance):
        # heatmap: (B,2,64,64) as-is
        # occupancy: (B,1,7,8) -> upsample to (B,1,64,64)
        occ_up = self.occupancy_upsample(occupancy)

        # impedance: (B,231) -> (B,self.imp_channels,8,8) -> upsample to (B,self.imp_channels,64,64)
        imp_proj = self.impedance_proj(impedance)
        imp_proj = imp_proj.view(-1, self.imp_channels, 8, 8)
        imp_up = self.impedance_upsample(imp_proj)

        fused = torch.cat([heatmap, occ_up, imp_up], dim=1)
        x = self.fusion_conv1(fused)
        x = self.fusion_conv2(x)
        x = self.attn_critic(x)
        x = self.fusion_conv3(x)
        feat = self.flatten(x)
        out = self.joint_fc(feat)
        return out

    def forward_with_intermediates(self, heatmap, occupancy, impedance):
        """
        Forward pass that returns intermediate features before fusion.
        
        Returns:
            out: Final critic score
            intermediates: Dict containing:
                - 'heatmap': (B, 2, 64, 64) - raw heatmap input
                - 'occ_up': (B, 1, 64, 64) - upsampled occupancy
                - 'imp_proj': (B, 4, 8, 8) - projected impedance (before upsample)
                - 'imp_up': (B, 4, 64, 64) - upsampled impedance
                - 'fused': (B, 7, 64, 64) - concatenated features before conv
                - 'after_conv1': (B, 16, 32, 32) - after first conv
                - 'after_conv2': (B, 32, 16, 16) - after second conv
                - 'after_attn': (B, 32, 16, 16) - after self-attention
                - 'after_conv3': (B, 64, 8, 8) - after third conv
        """
        # heatmap: (B,2,64,64) as-is
        # occupancy: (B,1,7,8) -> upsample to (B,1,64,64)
        occ_up = self.occupancy_upsample(occupancy)

        # impedance: (B,231) -> (B,self.imp_channels,8,8) -> upsample to (B,self.imp_channels,64,64)
        imp_proj = self.impedance_proj(impedance)
        imp_proj = imp_proj.view(-1, self.imp_channels, 8, 8)
        imp_up = self.impedance_upsample(imp_proj)

        fused = torch.cat([heatmap, occ_up, imp_up], dim=1)
        after_conv1 = self.fusion_conv1(fused)
        after_conv2 = self.fusion_conv2(after_conv1)
        after_attn = self.attn_critic(after_conv2)
        after_conv3 = self.fusion_conv3(after_attn)
        feat = self.flatten(after_conv3)
        out = self.joint_fc(feat)
        
        intermediates = {
            'heatmap': heatmap,
            'occ_up': occ_up,
            'imp_proj': imp_proj,
            'imp_up': imp_up,
            'fused': fused,
            'after_conv1': after_conv1,
            'after_conv2': after_conv2,
            'after_attn': after_attn,
            'after_conv3': after_conv3,
        }
        return out, intermediates
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for Conv/Linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                # If this is the final logit layer (out_features == 1), scale down to reduce early explosions
                if getattr(m, 'out_features', None) == 1:
                    with torch.no_grad():
                        m.weight.mul_(0.1)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)