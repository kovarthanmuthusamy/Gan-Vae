
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, cast


## Architecture for VAE with  multi-output (heatmap + occupancy + impedance + Max_Value)

Dr_value = 0.1  # Global dropout rate used throughout the model

# Encoder for multiple head /multi-modal VAE
class MultiInputVAE(nn.Module):
    def __init__(self, latent_dim=160,
                 heatmap_private_dim=48, occupancy_private_dim=12,
                 impedance_private_dim=32, maxvalue_private_dim=4,
                 shared_dim=64,
                 gumbel_temperature=0.5,
                 modality_dropout=0.5):
        """
        Multi-modal VAE with private and shared latent spaces.
        Private dim per modality scales with data complexity:
          - impedance (32): complex 231-point curve
          - heatmap   (48): 2-channel spatial map
          - occupancy (12): binary 52-vector
          - max_value  (4): single scalar

        Args:
            latent_dim: Total latent dim (default 160 = 48+12+32+4+64)
            heatmap_private_dim:   Private dim for heatmap   (default: 48)
            occupancy_private_dim: Private dim for occupancy (default: 12)
            impedance_private_dim: Private dim for impedance (default: 32)
            maxvalue_private_dim:  Private dim for max-value (default:  4)
            shared_dim: PoE shared latent dim               (default: 64)
        """
        super(MultiInputVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hm_priv   = heatmap_private_dim
        self.occ_priv  = occupancy_private_dim
        self.imp_priv  = impedance_private_dim
        self.mv_priv   = maxvalue_private_dim
        self.shared_dim = shared_dim
        self.temperature = gumbel_temperature  # Gumbel-Softmax temperature for occupancy
        self.modality_dropout = modality_dropout  # Prob of dropping each modality from PoE during training

        # Verify dimensions match
        total_private = heatmap_private_dim + occupancy_private_dim + impedance_private_dim + maxvalue_private_dim
        assert latent_dim == total_private + shared_dim, \
            f"latent_dim ({latent_dim}) must equal sum(private_dims)+shared_dim ({total_private + shared_dim})"

        # Heatmap head encoder branch
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # (16, 32, 32)
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
            # Input: (Batch, 1, 231) — 4-layer Conv1d with BatchNorm on all layers
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),   # 231 -> 116
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # 116 -> 58
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # 58 -> 29
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # 29 -> 15
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(128 * 15, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64)  # Final 64 features for Fusion
        )

        # Max Value encoder branch
        self.max_value_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU()
        )
       
        # Private latent space projections (dims scale with data complexity)
        self.heatmap_mu_private     = nn.Linear(64, heatmap_private_dim)
        self.heatmap_logvar_private = nn.Linear(64, heatmap_private_dim)

        # Occupancy uses Gumbel-Softmax: output logits for binary (0/1) per dim
        # Shape: (B, occ_priv, 2) — 2 logits per bit (score for 0 vs score for 1)
        self.occupancy_logits_private = nn.Linear(64, occupancy_private_dim * 2)

        self.impedance_mu_private     = nn.Linear(64, impedance_private_dim)   # larger: complex curve
        self.impedance_logvar_private = nn.Linear(64, impedance_private_dim)

        self.maxvalue_mu_private     = nn.Linear(64, maxvalue_private_dim)     # tiny: single scalar
        self.maxvalue_logvar_private = nn.Linear(64, maxvalue_private_dim)
        
        # Product of Experts: Each modality predicts shared latent space
        # These will be combined using PoE instead of simple concatenation
        self.heatmap_mu_shared = nn.Linear(64, shared_dim)
        self.heatmap_logvar_shared = nn.Linear(64, shared_dim)
        
        self.occupancy_mu_shared = nn.Linear(64, shared_dim)
        self.occupancy_logvar_shared = nn.Linear(64, shared_dim)
        
        self.impedance_mu_shared = nn.Linear(64, shared_dim)
        self.impedance_logvar_shared = nn.Linear(64, shared_dim)
        
        self.maxvalue_mu_shared = nn.Linear(64, shared_dim)
        self.maxvalue_logvar_shared = nn.Linear(64, shared_dim)

        # Per-decoder input sizes: each decoder receives [z_private_i || z_shared]
        hm_dec_in  = heatmap_private_dim  + shared_dim   # 16+64 = 80
        occ_dec_in = occupancy_private_dim + shared_dim  # 12+64 = 76
        imp_dec_in = impedance_private_dim + shared_dim  # 32+64 = 96
        mv_dec_in  = maxvalue_private_dim  + shared_dim  # 4+64  = 68

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
            nn.ConvTranspose2d(16, 2, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # Output in [0,1]
        )

        # Occupancy head decoder branch
        self.occupancy_decoder = nn.Sequential(
            nn.Linear(occ_dec_in, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),  # Project to spatial size with 512 channels
            nn.LeakyReLU(),
            nn.Linear(512,52),
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


        # Max Value decoder branch
        self.maxvalue_decoder = nn.Sequential(
            nn.Linear(mv_dec_in, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0,1]
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

    def gumbel_softmax_sample(self, logits, temperature, hard=False):
        """
        Sample from Gumbel-Softmax distribution.
        
        Args:
            logits: (B, occ_priv, 2) — unnormalized log-probs for each binary dim
            temperature: temperature τ (low → sharp 0/1, high → soft/blurry)
            hard: if True, use straight-through estimator (hard 0/1 in forward,
                  soft gradients in backward)
        Returns:
            z_occ: (B, occ_priv) — sampled values in [0,1] (or hard {0,1})
            probs: (B, occ_priv, 2) — softmax probabilities for KL computation
        """
        # Gumbel-Softmax: y_i = softmax((logit_i + g_i) / τ)
        y = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)  # (B, occ_priv, 2)
        
        # Take the probability of class 1 as the latent value
        z_occ = y[:, :, 1]  # (B, occ_priv)
        
        # Softmax probabilities (without Gumbel noise) for KL computation
        probs = F.softmax(logits, dim=-1)  # (B, occ_priv, 2)
        
        return z_occ, probs

    # forward method
    def encode(self, heatmap, occupancy, impedance, max_value, temperature=None):

        # Extract features from each branch
        heatmap_feat = self.heatmap_encoder(heatmap)  # (B, 64)
        occupancy_feat = self.occupancy_encoder(occupancy)  # (B, 64)
        
        # Impedance encoder: add channel dim for Conv1d
        impedance_feat = self.impedance_encoder(impedance.view(-1, 1, 231))  # (B, 64)
        
        max_value_feat = self.max_value_encoder(max_value)  # (B, 64)

        if temperature is None:
            temperature = self.temperature

        # Private latent spaces (modality-specific)
        # Gaussian private spaces (heatmap, impedance, maxvalue)
        heatmap_mu_priv = self.heatmap_mu_private(heatmap_feat)
        heatmap_logvar_priv = self.heatmap_logvar_private(heatmap_feat)
        
        impedance_mu_priv = self.impedance_mu_private(impedance_feat)
        impedance_logvar_priv = self.impedance_logvar_private(impedance_feat)
        
        maxvalue_mu_priv = self.maxvalue_mu_private(max_value_feat)
        maxvalue_logvar_priv = self.maxvalue_logvar_private(max_value_feat)
        
        # Gumbel-Softmax private space (occupancy — binary data)
        occ_logits = self.occupancy_logits_private(occupancy_feat)  # (B, occ_priv*2)
        occ_logits = occ_logits.view(-1, self.occ_priv, 2)  # (B, occ_priv, 2)
        
        # Sample occupancy latent via Gumbel-Softmax
        use_hard = not self.training  # hard samples at inference, soft during training
        z_occ_priv, occ_probs = self.gumbel_softmax_sample(occ_logits, temperature, hard=use_hard)
        
        # Product of Experts: Each modality predicts shared latent space
        heatmap_mu_shared = self.heatmap_mu_shared(heatmap_feat)
        heatmap_logvar_shared = self.heatmap_logvar_shared(heatmap_feat)
        
        occupancy_mu_shared = self.occupancy_mu_shared(occupancy_feat)
        occupancy_logvar_shared = self.occupancy_logvar_shared(occupancy_feat)
        
        impedance_mu_shared = self.impedance_mu_shared(impedance_feat)
        impedance_logvar_shared = self.impedance_logvar_shared(impedance_feat)
        
        maxvalue_mu_shared = self.maxvalue_mu_shared(max_value_feat)
        maxvalue_logvar_shared = self.maxvalue_logvar_shared(max_value_feat)
        
        # Modality dropout: during training, randomly replace some experts with prior
        # (mu=0, logvar=0 → σ²=1) so they contribute no information to PoE.
        # Forces each modality to independently encode enough into shared space.
        mu_shared_list = [heatmap_mu_shared, occupancy_mu_shared, impedance_mu_shared, maxvalue_mu_shared]
        logvar_shared_list = [heatmap_logvar_shared, occupancy_logvar_shared, impedance_logvar_shared, maxvalue_logvar_shared]

        if self.training and self.modality_dropout > 0:
            # Sample a drop mask, but ensure at least one modality survives
            keep = [torch.rand(1).item() > self.modality_dropout for _ in range(4)]
            if not any(keep):  # all dropped → keep a random one
                keep[int(torch.randint(4, (1,)))] = True
            prior_mu = torch.zeros_like(mu_shared_list[0])
            prior_lv = torch.zeros_like(logvar_shared_list[0])
            for i in range(4):
                if not keep[i]:
                    mu_shared_list[i] = prior_mu
                    logvar_shared_list[i] = prior_lv

        # Combine shared predictions using Product of Experts
        mu_shared, logvar_shared = self.product_of_experts(
            mu_shared_list, logvar_shared_list
        )
        
        # --- Gaussian private: concatenate mu/logvar for heatmap, impedance, maxvalue ---
        mu_gauss_private = torch.cat([heatmap_mu_priv, 
                                      impedance_mu_priv, maxvalue_mu_priv], dim=1)
        logvar_gauss_private = torch.cat([heatmap_logvar_priv, 
                                          impedance_logvar_priv, maxvalue_logvar_priv], dim=1)
        
        # Reparameterize Gaussian private + shared
        mu_gaussian = torch.cat([mu_gauss_private, mu_shared], dim=1)
        logvar_gaussian = torch.cat([logvar_gauss_private, logvar_shared], dim=1)
        
        z_gaussian = self._reparameterize_gaussian(mu_gaussian, logvar_gaussian)
        
        # --- Build full z: [hm_priv | occ_priv(gumbel) | imp_priv | mv_priv | shared] ---
        # Split z_gaussian back into parts for correct layout
        hm_size = self.hm_priv
        imp_size = self.imp_priv
        mv_size = self.mv_priv
        shared_size = self.shared_dim
        
        z_hm = z_gaussian[:, :hm_size]
        z_imp = z_gaussian[:, hm_size:hm_size+imp_size]
        z_mv = z_gaussian[:, hm_size+imp_size:hm_size+imp_size+mv_size]
        z_shared = z_gaussian[:, hm_size+imp_size+mv_size:]
        
        z = torch.cat([z_hm, z_occ_priv, z_imp, z_mv, z_shared], dim=1)  # (B, latent_dim)
        
        # Store per-modality statistics for monitoring
        modality_stats = {
            'heatmap':   (heatmap_mu_priv,   heatmap_logvar_priv),
            'occupancy': (occ_logits, occ_probs),  # logits and probs for Gumbel
            'impedance': (impedance_mu_priv, impedance_logvar_priv),
            'maxvalue':  (maxvalue_mu_priv,  maxvalue_logvar_priv)
        }
        
        return z, mu_gaussian, logvar_gaussian, occ_logits, modality_stats
        
    def _reparameterize_gaussian(self, mu, logvar):
        """Gaussian reparameterization trick"""
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, device=std.device)
        return mu + eps * std

    # Decoder method forward pass  
    def decode(self, z):

        # Slice z into per-modality private portions + shared space
        # Layout: [hm_priv | occ_priv | imp_priv | mv_priv | shared]
        z_hm     = z[:, :self.hm_priv]
        z_occ    = z[:, self.hm_priv : self.hm_priv + self.occ_priv]
        z_imp    = z[:, self.hm_priv + self.occ_priv : self.hm_priv + self.occ_priv + self.imp_priv]
        z_mv     = z[:, self.hm_priv + self.occ_priv + self.imp_priv : -self.shared_dim]
        z_shared = z[:, -self.shared_dim:]

        # Each decoder sees [z_private_i || z_shared]: modality-specific detail + global context
        heatmap_in = torch.cat([z_hm,  z_shared], dim=1)
        occ_in     = torch.cat([z_occ, z_shared], dim=1)
        imp_in     = torch.cat([z_imp, z_shared], dim=1)
        mv_in      = torch.cat([z_mv,  z_shared], dim=1)

        # Decode heatmap
        heatmap_feat = self.heatmap_fc(heatmap_in)
        heatmap_feat = heatmap_feat.view(-1, 64, 8, 8)
        heatmap_recon = self.heatmap_deconv(heatmap_feat)  # (B, 2, 64, 64)

        # Decode occupancy
        occupancy_recon = self.occupancy_decoder(occ_in)   # (B, 52)

        # Decode impedance: pure FC
        impedance_recon = self.impedance_decoder(imp_in)    # (B, 231)

        # Decode max value
        max_value_recon = self.maxvalue_decoder(mv_in)     # (B, 1)

        return heatmap_recon, occupancy_recon, impedance_recon, max_value_recon


    # full forward pass   
    def forward(self, heatmap, occupancy, impedance, max_value, temperature=None):
        """
        Full forward pass through encoder and decoder.
        
        Returns:
            heatmap_recon, occupancy_recon, impedance_recon, max_value_recon,
            mu_gaussian, logvar_gaussian,  # Gaussian parts only (for Gaussian KL)
            occ_logits,                    # Occupancy Gumbel logits (for Gumbel KL)
            modality_stats                 # Per-modality monitoring stats
        """
        # Encode (reparameterization now happens inside encode)
        z, mu_gaussian, logvar_gaussian, occ_logits, modality_stats = self.encode(
            heatmap, occupancy, impedance, max_value, temperature
        )
        
        # Decode
        heatmap_recon, occupancy_recon, impedance_recon, max_value_recon = self.decode(z)
        
        return (heatmap_recon, occupancy_recon, impedance_recon, max_value_recon,
                mu_gaussian, logvar_gaussian, occ_logits, modality_stats)

    ## for inference or sampling
    def inference(self, num_samples, device, latent_stats=None):
        """
        Generate samples from random latent vectors.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            latent_stats: Optional dict with per-modality sampling parameters.
                Keys: 'heatmap', 'impedance', 'maxvalue', 'shared', 'occupancy'
                Each value is a dict with 'mu_mean', 'mu_std' (floats or 1-D tensors).
                If None, samples from N(0,1) for Gaussian dims.
        """
        with torch.no_grad():
            if latent_stats is None:
                # Fallback: standard normal for all Gaussian dims
                stats = {k: {'mu_mean': 0.0, 'mu_std': 1.0}
                         for k in ('heatmap', 'impedance', 'maxvalue', 'shared')}
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
            z_mv = _sample('maxvalue', self.mv_priv)
            z_shared = _sample('shared', self.shared_dim)

            # Occupancy part: sample binary from Bernoulli(0.5)
            z_occ = torch.bernoulli(torch.full((num_samples, self.occ_priv), 0.5)).to(device)
            
            # Assemble: [hm_priv | occ_priv | imp_priv | mv_priv | shared]
            z = torch.cat([z_hm, z_occ, z_imp, z_mv, z_shared], dim=1)
            
            heatmap, occupancy, impedance, max_value = self.decode(z)
            return heatmap, occupancy, impedance, max_value
