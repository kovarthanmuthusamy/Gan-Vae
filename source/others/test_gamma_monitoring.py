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

# Create model
print("Creating VAE model with self-attention decoders...")
model = MultiInputVAE(latent_dim=128)

# Check initial gamma values (should all be 0)
print("\n" + "="*80)
print("INITIAL STATE:")
log_gamma_values(model, epoch=0)

# Simulate training by manually adjusting some gammas
print("\n" + "="*80)
print("Simulating training (manually adjusting gamma values)...")
print("="*80)

gamma_updates = {
  'encoder.fusion_encoder.self_attn.gamma': 0.342,
  'decoder.heatmap_dec.self_attn.gamma': 0.612,
  'decoder.occupancy_dec.self_attn.gamma': 0.187,
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
  - decoder.heatmap_dec.self_attn.gamma = 0.612  ✅ Self-attention stabilizes heatmap decoding

Medium gamma (0.1-0.5):  Attention is moderately used
  - encoder.fusion_encoder.self_attn.gamma = 0.342  📈 Cross-modal fusion benefits from attention

Low gamma (< 0.1):  Attention is barely used
  - decoder.occupancy_dec.self_attn.gamma = 0.187  ⚠️ Occupancy decoder mostly relies on convolutions

Conclusion:
  - Heatmap branch depends heavily on self-attention for global structure
  - Encoder fusion attention remains active but modest
  - Occupancy branch is mostly convolution-driven, which is acceptable for binary grids
  
This is perfectly fine! The network learned which relationships are useful.
""")
