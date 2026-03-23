
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, cast


## Architecture for VAE with multi-output (heatmap + occupancy + impedance)
## Heatmap: 1-channel log(1+x) z-score normalized (64x64)
## Background pixels get a fixed z-score value, no separate mask channel needed

Dr_value = 0.1  # Global dropout rate used throughout the model


class SE1d(nn.Module):
    """
    Squeeze-and-Excitation block for 1D feature maps (B, C, L).

    Squeeze:  global average pool over the temporal axis → (B, C)
    Excite:   two-layer FC bottleneck (C → C//reduction → C) + Sigmoid → (B, C)
    Scale:    broadcast-multiply attention weights back onto the feature map.

    Lets the impedance encoder re-weight frequency-band channels
    adaptively before the next conv, emphasising resonance-relevant bands.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        s = x.mean(dim=2)           # Squeeze:  (B, C)
        s = self.fc(s)              # Excite:   (B, C)
        return x * s.unsqueeze(2)   # Scale:    (B, C, L)


# Encoder for multiple head /multi-modal VAE
class MultiInputVAE(nn.Module):
    def __init__(self, latent_dim=132,
                 heatmap_private_dim=32, occupancy_private_dim=32,
                 impedance_private_dim=20,
                 shared_dim=48,
                 gumbel_temperature=0.5,
                 modality_dropout=0.5):
        """
        Multi-modal VAE with private and shared latent spaces.
        3 modalities: heatmap (1ch z-score), occupancy (binary), impedance (z-score).

        Private dim per modality scales with data complexity:
          - heatmap   (48): 1-channel spatial map (log(1+x) z-score)
          - impedance (32): complex 231-point curve
          - occupancy (32): binary 52-vector (needs enough binary codes)

        Args:
            latent_dim: Total latent dim (default 160 = 48+32+32+48)
            heatmap_private_dim:   Private dim for heatmap   (default: 48)
            occupancy_private_dim: Private dim for occupancy (default: 32)
            impedance_private_dim: Private dim for impedance (default: 32)
            shared_dim: PoE shared latent dim               (default: 48)
        """
        super(MultiInputVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hm_priv   = heatmap_private_dim
        self.occ_priv  = occupancy_private_dim
        self.imp_priv  = impedance_private_dim
        self.shared_dim = shared_dim
        self.temperature = gumbel_temperature  # Gumbel-Softmax temperature for occupancy
        self.modality_dropout = modality_dropout  # Prob of dropping each modality from PoE during training

        # Verify dimensions match
        total_private = heatmap_private_dim + occupancy_private_dim + impedance_private_dim
        assert latent_dim == total_private + shared_dim, \
            f"latent_dim ({latent_dim}) must equal sum(private_dims)+shared_dim ({total_private + shared_dim})"

        # Heatmap head encoder branch (1-channel: log(1+x) z-score)
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (16, 32, 32)
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 8, 8)
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512,64),
            nn.Dropout(Dr_value),
            nn.LeakyReLU()
        )
        # Occupancy head encoder branch (increased complexity)
        self.occupancy_encoder = nn.Sequential(
            # First conv block
            nn.Linear(52,128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),  # Output 64 features to match other branches
            nn.LeakyReLU()
        )

        self.impedance_encoder = nn.Sequential(
            # Input: (Batch, 1, 231) — 4-layer Conv1d + SE blocks for channel re-weighting
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),   # 231 -> 116
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            SE1d(16, reduction=4),                                   # SE: focus on dominant bands
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # 116 -> 58
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            SE1d(32, reduction=4),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # 58 -> 29
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            SE1d(64, reduction=4),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # 29 -> 15
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            SE1d(128, reduction=4),
            nn.Flatten(),
            nn.Linear(128 * 15, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64)  # Final 64 features for Fusion
        )

        # Private latent space projections (dims scale with data complexity)
        self.heatmap_mu_private     = nn.Linear(64, heatmap_private_dim)
        self.heatmap_logvar_private = nn.Linear(64, heatmap_private_dim)

        # Occupancy uses Binary Concrete (relaxed Bernoulli): 1 logit per dim
        # sigmoid(logit) → probability of bit=1, sampled via concrete relaxation
        self.occupancy_logits_private = nn.Linear(64, occupancy_private_dim)

        self.impedance_mu_private     = nn.Linear(64, impedance_private_dim)
        self.impedance_logvar_private = nn.Linear(64, impedance_private_dim)
        
        # Product of Experts: Each modality predicts shared latent space
        # These will be combined using PoE instead of simple concatenation
        self.heatmap_mu_shared = nn.Linear(64, shared_dim)
        self.heatmap_logvar_shared = nn.Linear(64, shared_dim)
        
        self.occupancy_mu_shared = nn.Linear(64, shared_dim)
        self.occupancy_logvar_shared = nn.Linear(64, shared_dim)
        
        self.impedance_mu_shared = nn.Linear(64, shared_dim)
        self.impedance_logvar_shared = nn.Linear(64, shared_dim)

        # Per-decoder input sizes: each decoder receives [z_private_i || z_shared]
        hm_dec_in  = heatmap_private_dim  + shared_dim   # 48+48 = 96
        occ_dec_in = occupancy_private_dim + shared_dim  # 32+48 = 80
        imp_dec_in = impedance_private_dim + shared_dim  # 32+48 = 80

        # Heatmap head decoder branch
        self.heatmap_fc = nn.Sequential(
            nn.Linear(hm_dec_in, 256),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(256, 64 * 8 * 8),
            nn.BatchNorm1d(64 * 8 * 8),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value)
        )
        self.heatmap_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            # No activation: output is log(1+x) z-score normalized
        )

        # Occupancy head decoder branch — deeper to handle binary latent inputs
        # Binary codes need more expressive mapping than continuous latents
        self.occupancy_decoder = nn.Sequential(
            nn.Linear(occ_dec_in, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 52),
            nn.Sigmoid()  # (B, 52)
        )

        # Impedance head decoder branch — pure FC (smooth data: ~4 peaks, ~4 valleys)
        # No ConvTranspose1d → no checkerboard artifacts
        self.impedance_decoder = nn.Sequential(
            nn.Linear(imp_dec_in, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 231),
            # No activation: output is z-score normalized
        )

    def product_of_experts(self, mu_list, logvar_list, eps=1e-8):
        """
        Product of Experts: Combine multiple Gaussian distributions
        
        For N experts with (μ_i, σ²_i), the combined distribution is:
        - Precision: 1/σ²_combined = Σ(1/σ²_i)
        - Mean: μ_combined = σ²_combined * Σ(μ_i/σ²_i)
        
        Args:
            mu_list: List of mean tensors from different modalities
            logvar_list: List of logvar tensors from different modalities
            eps: Small constant for numerical stability
            
        Returns:
            combined_mu: Combined mean
            combined_logvar: Combined log variance
        """
        # Add unit-Gaussian prior expert (mu=0, sigma²=1, logvar=0) as shared-space anchor.
        # This prevents high-confidence simple modalities (e.g. scalar max_value)
        # from dominating the PoE sum and drowning out complex ones (e.g. impedance curve).
        prior_mu     = torch.zeros_like(mu_list[0])
        prior_logvar = torch.zeros_like(logvar_list[0])  # sigma²=exp(0)=1
        mu_list      = mu_list     + [prior_mu]
        logvar_list  = logvar_list + [prior_logvar]

        # Convert logvar to variance
        var_list = [torch.exp(logvar) + eps for logvar in logvar_list]
        
        # Compute combined precision (sum of precisions)
        # 1/σ²_combined = Σ(1/σ²_i)
        precision_list = [1.0 / var for var in var_list]
        combined_precision = torch.stack(precision_list).sum(dim=0)
        
        # Compute combined variance
        combined_var = 1.0 / (combined_precision + eps)
        
        # Compute combined mean
        # μ_combined = σ²_combined * Σ(μ_i/σ²_i)
        weighted_mu_list = [mu * precision for mu, precision in zip(mu_list, precision_list)]
        combined_mu = combined_var * torch.stack(weighted_mu_list).sum(dim=0)
        
        # Convert back to logvar
        combined_logvar = torch.log(combined_var + eps)
        
        return combined_mu, combined_logvar

    def binary_concrete_sample(self, logits, temperature):
        """
        Sample from Binary Concrete (relaxed Bernoulli) distribution.
        
        z = sigmoid((logit + log(u/(1-u))) / τ)  where u ~ Uniform(0,1)
        As τ→0, z → Bernoulli(sigmoid(logit)).
        Smooth gradients for all τ > 0.
        
        Args:
            logits: (B, occ_priv) — log-odds for each binary dim
            temperature: τ > 0 (low → sharp, high → soft)
        Returns:
            z_occ: (B, occ_priv) — sampled values in (0,1)
            probs: (B, occ_priv) — sigmoid(logits), P(bit=1) for KL
        """
        probs = torch.sigmoid(logits)  # P(bit=1) per dim
        
        if self.training:
            # Binary Concrete: add logistic noise scaled by temperature
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            noise = torch.log(u / (1 - u))  # Logistic noise
            z_occ = torch.sigmoid((logits + noise) / temperature)  # (B, occ_priv)
        else:
            # At inference: hard threshold at 0.5
            z_occ = (probs > 0.5).float()
        
        return z_occ, probs

    # forward method
    def encode(self, heatmap, occupancy, impedance, temperature=None):

        # Extract features from each branch
        heatmap_feat = self.heatmap_encoder(heatmap)  # (B, 64)
        occupancy_feat = self.occupancy_encoder(occupancy)  # (B, 64)
        
        # Impedance encoder: add channel dim for Conv1d
        impedance_feat = self.impedance_encoder(impedance.view(-1, 1, 231))  # (B, 64)

        if temperature is None:
            temperature = self.temperature

        # Private latent spaces (modality-specific)
        # Gaussian private spaces (heatmap, impedance)
        heatmap_mu_priv = self.heatmap_mu_private(heatmap_feat)
        heatmap_logvar_priv = self.heatmap_logvar_private(heatmap_feat)
        
        impedance_mu_priv = self.impedance_mu_private(impedance_feat)
        impedance_logvar_priv = self.impedance_logvar_private(impedance_feat)
        
        # Binary Concrete private space (occupancy — binary data)
        occ_logits = self.occupancy_logits_private(occupancy_feat)  # (B, occ_priv)
        
        # Sample occupancy latent via Binary Concrete (smooth gradients)
        z_occ_priv, occ_probs = self.binary_concrete_sample(occ_logits, temperature)
        
        # Product of Experts: Each modality predicts shared latent space
        heatmap_mu_shared = self.heatmap_mu_shared(heatmap_feat)
        heatmap_logvar_shared = self.heatmap_logvar_shared(heatmap_feat)
        
        occupancy_mu_shared = self.occupancy_mu_shared(occupancy_feat)
        occupancy_logvar_shared = self.occupancy_logvar_shared(occupancy_feat)
        
        impedance_mu_shared = self.impedance_mu_shared(impedance_feat)
        impedance_logvar_shared = self.impedance_logvar_shared(impedance_feat)
        
        # Modality dropout: during training, randomly replace some experts with prior
        # (mu=0, logvar=0 → σ²=1) so they contribute no information to PoE.
        # Forces each modality to independently encode enough into shared space.
        mu_shared_list = [heatmap_mu_shared, occupancy_mu_shared, impedance_mu_shared]
        logvar_shared_list = [heatmap_logvar_shared, occupancy_logvar_shared, impedance_logvar_shared]

        if self.training and self.modality_dropout > 0:
            # Sample a drop mask, but ensure at least one modality survives
            keep = [torch.rand(1).item() > self.modality_dropout for _ in range(3)]
            if not any(keep):  # all dropped → keep a random one
                keep[int(torch.randint(3, (1,)))] = True
            prior_mu = torch.zeros_like(mu_shared_list[0])
            prior_lv = torch.zeros_like(logvar_shared_list[0])
            for i in range(3):
                if not keep[i]:
                    mu_shared_list[i] = prior_mu
                    logvar_shared_list[i] = prior_lv

        # Combine shared predictions using Product of Experts
        mu_shared, logvar_shared = self.product_of_experts(
            mu_shared_list, logvar_shared_list
        )
        
        # --- Gaussian private: concatenate mu/logvar for heatmap, impedance ---
        mu_gauss_private = torch.cat([heatmap_mu_priv, impedance_mu_priv], dim=1)
        logvar_gauss_private = torch.cat([heatmap_logvar_priv, impedance_logvar_priv], dim=1)

        # Clamp private logvars: [-4, 2] → σ_floor=0.135
        # Clamp shared logvar: [-2, 2] → σ_floor=0.368
        # Looser shared floor gives the PoE combined distribution room to rise above the
        # precision-sum floor (3 experts at σ=0.135 → combined σ≈0.078, which always
        # gets clamped back up and prevents the shared space from learning real variance).
        logvar_gauss_private = torch.clamp(logvar_gauss_private, min=-4.0, max=2.0)
        logvar_shared        = torch.clamp(logvar_shared,        min=-2.0, max=2.0)

        # Reparameterize Gaussian private + shared
        mu_gaussian     = torch.cat([mu_gauss_private, mu_shared],          dim=1)
        logvar_gaussian = torch.cat([logvar_gauss_private, logvar_shared],  dim=1)
        
        z_gaussian = self._reparameterize_gaussian(mu_gaussian, logvar_gaussian)
        
        # --- Build full z: [hm_priv | occ_priv(gumbel) | imp_priv | shared] ---
        hm_size = self.hm_priv
        imp_size = self.imp_priv
        
        z_hm = z_gaussian[:, :hm_size]
        z_imp = z_gaussian[:, hm_size:hm_size+imp_size]
        z_shared = z_gaussian[:, hm_size+imp_size:]
        
        z = torch.cat([z_hm, z_occ_priv, z_imp, z_shared], dim=1)  # (B, latent_dim)
        
        # Store per-modality statistics for monitoring (post-clamp logvars for accurate σ reporting)
        logvar_hm_clamped  = logvar_gaussian[:, :hm_size]
        logvar_imp_clamped = logvar_gaussian[:, hm_size:hm_size + imp_size]
        modality_stats = {
            'heatmap':   (heatmap_mu_priv,   logvar_hm_clamped),
            'occupancy': (occ_logits, occ_probs),  # logits (B, occ_priv) and probs (B, occ_priv)
            'impedance': (impedance_mu_priv, logvar_imp_clamped),
        }
        
        return z, mu_gaussian, logvar_gaussian, occ_logits, modality_stats
        
    def _reparameterize_gaussian(self, mu, logvar):
        """Gaussian reparameterization trick. Input logvar should already be clamped at call site;
        this clamp is a safety net to prevent NaN in exp() from any unexpected raw logvar path."""
        logvar = torch.clamp(logvar, min=-4, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, device=std.device)
        return mu + eps * std

    # Decoder method forward pass  
    def decode(self, z):

        # Slice z into per-modality private portions + shared space
        # Layout: [hm_priv | occ_priv | imp_priv | shared]
        z_hm     = z[:, :self.hm_priv]
        z_occ    = z[:, self.hm_priv : self.hm_priv + self.occ_priv]
        z_imp    = z[:, self.hm_priv + self.occ_priv : self.hm_priv + self.occ_priv + self.imp_priv]
        z_shared = z[:, -self.shared_dim:]

        # Each decoder sees [z_private_i || z_shared]: modality-specific detail + global context
        heatmap_in = torch.cat([z_hm,  z_shared], dim=1)
        occ_in     = torch.cat([z_occ, z_shared], dim=1)
        imp_in     = torch.cat([z_imp, z_shared], dim=1)

        # Decode heatmap (1-channel z-score)
        heatmap_feat = self.heatmap_fc(heatmap_in)
        heatmap_feat = heatmap_feat.view(-1, 64, 8, 8)
        heatmap_recon = self.heatmap_deconv(heatmap_feat)  # (B, 1, 64, 64)

        # Decode occupancy
        occupancy_recon = self.occupancy_decoder(occ_in)   # (B, 52)

        # Decode impedance: pure FC
        impedance_recon = self.impedance_decoder(imp_in)    # (B, 231)

        return heatmap_recon, occupancy_recon, impedance_recon

    def encode_cross_modal(self, source: str, heatmap=None, impedance=None, temperature=None):
        """
        Encode using a SINGLE source modality for cross-modal training.

        Source modality: its private encoder + its shared expert are used.
        Non-source private dims: zeros (prior mean — no gradient signal from missing input).
        Occupancy: sampled from prior Bernoulli(0.5) (logits=0).
        Shared PoE: only the source expert active; product_of_experts adds prior anchor internally.

        Used to train the shared space to carry cross-modal information:
          loss = recon(decode(z_cross), all_targets) forces shared dims to represent
          enough structure for any decoder to reconstruct from any single input.

        Args:
            source: 'heatmap' or 'impedance'
            heatmap: (B, 1, H, W) — required when source='heatmap'
            impedance: (B, 231) — required when source='impedance'
            temperature: Gumbel temperature for occupancy prior sampling
        Returns:
            z: (B, latent_dim) ready for decode()
        """
        if temperature is None:
            temperature = self.temperature

        B      = heatmap.shape[0] if heatmap is not None else impedance.shape[0] # type: ignore
        device = heatmap.device   if heatmap is not None else impedance.device # type: ignore

        # Occupancy prior: logits=0 → P(1)=0.5, no information
        z_occ_priv, _ = self.binary_concrete_sample(
            torch.zeros(B, self.occ_priv, device=device), temperature
        )

        if source == 'heatmap':
            feat = self.heatmap_encoder(heatmap)
            z_hm_priv = self._reparameterize_gaussian(
                self.heatmap_mu_private(feat),
                torch.clamp(self.heatmap_logvar_private(feat), -4.0, 2.0)
            )
            z_imp_priv = torch.zeros(B, self.imp_priv, device=device)
            mu_sh = self.heatmap_mu_shared(feat)
            lv_sh = self.heatmap_logvar_shared(feat)

        elif source == 'impedance':
            feat = self.impedance_encoder(impedance.view(-1, 1, 231)) # type: ignore
            z_hm_priv  = torch.zeros(B, self.hm_priv, device=device)
            z_imp_priv = self._reparameterize_gaussian(
                self.impedance_mu_private(feat),
                torch.clamp(self.impedance_logvar_private(feat), -4.0, 2.0)
            )
            mu_sh = self.impedance_mu_shared(feat)
            lv_sh = self.impedance_logvar_shared(feat)

        else:
            raise ValueError(f"Unsupported source: {source!r}. Use 'heatmap' or 'impedance'.")

        # PoE with single expert (product_of_experts appends the prior anchor automatically)
        # Use shared floor of -2.0 (σ=0.368) to match encode() shared clamp
        mu_shared, logvar_shared = self.product_of_experts([mu_sh], [lv_sh])
        z_shared = self._reparameterize_gaussian(
            mu_shared, torch.clamp(logvar_shared, -2.0, 2.0)
        )

        return torch.cat([z_hm_priv, z_occ_priv, z_imp_priv, z_shared], dim=1)

    # full forward pass   
    def forward(self, heatmap, occupancy, impedance, temperature=None):
        """
        Full forward pass through encoder and decoder.
        
        Returns:
            heatmap_recon, occupancy_recon, impedance_recon,
            mu_gaussian, logvar_gaussian,  # Gaussian parts only (for Gaussian KL)
            occ_logits,                    # Occupancy Gumbel logits (for Gumbel KL)
            modality_stats                 # Per-modality monitoring stats
        """
        # Encode (reparameterization now happens inside encode)
        z, mu_gaussian, logvar_gaussian, occ_logits, modality_stats = self.encode(
            heatmap, occupancy, impedance, temperature
        )
        
        # Decode
        heatmap_recon, occupancy_recon, impedance_recon = self.decode(z)
        
        return (heatmap_recon, occupancy_recon, impedance_recon,
                mu_gaussian, logvar_gaussian, occ_logits, modality_stats)

    ## for inference or sampling
    def inference(self, num_samples, device, latent_stats=None):
        """
        Generate samples from random latent vectors.
        3 modalities: heatmap, occupancy (Gumbel), impedance.
        """
        with torch.no_grad():
            if latent_stats is None:
                # Fallback: standard normal for all Gaussian dims
                stats = {k: {'mu_mean': 0.0, 'mu_std': 1.0}
                         for k in ('heatmap', 'impedance', 'shared')}
            else:
                stats = latent_stats

            def _sample(name, dim):
                mu = stats[name].get('mu_mean', 0.0)
                mu_std = stats[name].get('mu_std', 1.0)
                sigma_mean = stats[name].get('sigma_mean', 0.0)
                # Aggregate posterior std: sqrt(Var[mu] + E[sigma^2])
                agg_std = (mu_std**2 + sigma_mean**2) ** 0.5
                return (torch.randn(num_samples, dim).to(device) * agg_std + mu)

            z_hm = _sample('heatmap', self.hm_priv)
            z_imp = _sample('impedance', self.imp_priv)
            z_shared = _sample('shared', self.shared_dim)

            # Occupancy part: sample binary from Bernoulli(0.5)
            z_occ = torch.bernoulli(torch.full((num_samples, self.occ_priv), 0.5)).to(device)
            
            # Assemble: [hm_priv | occ_priv | imp_priv | shared]
            z = torch.cat([z_hm, z_occ, z_imp, z_shared], dim=1)
            
            heatmap, occupancy, impedance = self.decode(z)
            return heatmap, occupancy, impedance
