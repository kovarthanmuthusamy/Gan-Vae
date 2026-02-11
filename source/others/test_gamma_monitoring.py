#!/usr/bin/env python3
"""
Quick test to demonstrate gamma monitoring in action
"""

import torch
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "source"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from model.vae_multi_input import MultiInputVAE
from monitor_attention_gammas import log_gamma_values  # noqa: E402

# Create model with cross-attention enabled
print("Creating VAE model with cross-attention...")
model = MultiInputVAE(latent_dim=128, use_cross_attn=True)

# Check initial gamma values (should all be 0)
print("\n" + "="*80)
print("INITIAL STATE:")
log_gamma_values(model, epoch=0)

# Simulate training by manually adjusting some gammas
print("\n" + "="*80)
print("Simulating training (manually adjusting gamma values)...")
print("="*80)

gamma_updates = {
    'encoder.heatmap_enc.self_attn.gamma': 0.342,
    'encoder.occupancy_enc.self_attn.gamma': 0.187,
    'encoder.impedance_enc.self_attn.gamma': 0.098,
    'decoder.heatmap_dec.self_attn.gamma': 0.612,
    'decoder.heatmap_imp_attn.gamma': 0.891,
    'decoder.heatmap_occ_attn.gamma': 0.043,
    'decoder.occupancy_dec.self_attn.gamma': 0.421,
    'decoder.occ_imp_attn.gamma': 0.678,
    'decoder.occ_heat_attn.gamma': 0.234,
}

for name, new_value in gamma_updates.items():
    for param_name, param in model.named_parameters():
        if param_name == name:
            param.data.fill_(new_value)
            print(f"  Set {param_name} = {new_value}")

# Check gamma values after "training"
print("\n" + "="*80)
print("AFTER SIMULATED TRAINING (Epoch 100):")
log_gamma_values(model, epoch=100)

# Summary
print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("""
High gamma (> 0.5):  Attention is strongly used
  - decoder.heatmap_imp_attn.gamma = 0.891  ✅ Heatmap benefits from impedance
  - decoder.occ_imp_attn.gamma = 0.678      ✅ Occupancy benefits from impedance
  - decoder.heatmap_dec.self_attn.gamma = 0.612  ✅ Self-attention helps

Medium gamma (0.1-0.5):  Attention is moderately used
  - decoder.occupancy_dec.self_attn.gamma = 0.421  📈 Some benefit
  - decoder.occ_heat_attn.gamma = 0.234           📈 Some benefit

Low gamma (< 0.1):  Attention is barely used
  - decoder.heatmap_occ_attn.gamma = 0.043  ⚠️ Heatmap doesn't need occupancy info
  - encoder.impedance_enc.self_attn.gamma = 0.098  ⚠️ Weak self-attention

Conclusion:
  - Impedance ↔ Heatmap relationship is STRONG
  - Impedance ↔ Occupancy relationship is STRONG  
  - Heatmap ↔ Occupancy relationship is WEAK (heatmap doesn't need occupancy)
  
This is perfectly fine! The network learned which relationships are useful.
""")
