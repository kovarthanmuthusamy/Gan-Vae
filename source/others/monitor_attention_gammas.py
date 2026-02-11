"""
Monitor gamma parameters during training to see if attention layers are active

Usage:
    # In your training loop:
    from scripts.monitor_attention_gammas import log_gamma_values
    
    if epoch % 10 == 0:
        log_gamma_values(model, epoch)
"""

import torch
import torch.nn as nn
from typing import Optional


def log_gamma_values(model: nn.Module, epoch: Optional[int] = None):
    """
    Print all gamma parameters in the model to see which attention layers are active
    
    Args:
        model: The VAE model
        epoch: Optional epoch number for logging
    """
    print("\n" + "="*80)
    if epoch is not None:
        print(f"GAMMA VALUES AT EPOCH {epoch}")
    else:
        print("GAMMA VALUES")
    print("="*80)
    
    gamma_dict = {}
    
    for name, param in model.named_parameters():
        if 'gamma' in name:
            gamma_value = param.item()
            gamma_dict[name] = gamma_value
    
    # Group by module type
    encoder_gammas = {k: v for k, v in gamma_dict.items() if 'encoder' in k}
    decoder_gammas = {k: v for k, v in gamma_dict.items() if 'decoder' in k}
    
    print("\n📊 ENCODER ATTENTION GAMMAS:")
    print("-" * 80)
    for name, value in sorted(encoder_gammas.items()):
        status = get_gamma_status(value)
        print(f"  {name:60s} = {value:+.6f}  {status}")
    
    print("\n📊 DECODER ATTENTION GAMMAS:")
    print("-" * 80)
    for name, value in sorted(decoder_gammas.items()):
        status = get_gamma_status(value)
        print(f"  {name:60s} = {value:+.6f}  {status}")
    
    # Summary statistics
    all_gammas = list(gamma_dict.values())
    if all_gammas:
        print("\n📈 SUMMARY:")
        print("-" * 80)
        print(f"  Total gamma parameters: {len(all_gammas)}")
        print(f"  Mean: {sum(all_gammas)/len(all_gammas):.6f}")
        print(f"  Max:  {max(all_gammas):+.6f}")
        print(f"  Min:  {min(all_gammas):+.6f}")
        
        # Count by status
        active = sum(1 for v in all_gammas if abs(v) > 0.1)
        learning = sum(1 for v in all_gammas if 0.01 < abs(v) <= 0.1)
        inactive = sum(1 for v in all_gammas if abs(v) <= 0.01)
        
        print(f"\n  Active (|γ| > 0.1):      {active}/{len(all_gammas)}")
        print(f"  Learning (0.01 < |γ| ≤ 0.1): {learning}/{len(all_gammas)}")
        print(f"  Inactive (|γ| ≤ 0.01):    {inactive}/{len(all_gammas)}")
    
    print("="*80 + "\n")
    
    return gamma_dict


def get_gamma_status(gamma: float) -> str:
    """Get human-readable status for gamma value"""
    abs_gamma = abs(gamma)
    
    if abs_gamma > 1.0:
        return "🔥 VERY ACTIVE (attention dominates)"
    elif abs_gamma > 0.5:
        return "✅ ACTIVE (attention significant)"
    elif abs_gamma > 0.1:
        return "📈 LEARNING (attention emerging)"
    elif abs_gamma > 0.01:
        return "🌱 WEAK (attention minimal)"
    else:
        return "❌ INACTIVE (attention off)"


def get_attention_usage_summary(model: nn.Module) -> dict:
    """
    Get a summary of which attention layers are actually being used
    
    Returns:
        dict with counts of active/inactive attention layers
    """
    gamma_values = []
    
    for name, param in model.named_parameters():
        if 'gamma' in name:
            gamma_values.append((name, param.item()))
    
    summary = {
        'total': len(gamma_values),
        'active': sum(1 for _, v in gamma_values if abs(v) > 0.1),
        'learning': sum(1 for _, v in gamma_values if 0.01 < abs(v) <= 0.1),
        'inactive': sum(1 for _, v in gamma_values if abs(v) <= 0.01),
        'values': dict(gamma_values)
    }
    
    return summary


def check_attention_gradients(model: nn.Module):
    """
    Check if gamma parameters are receiving meaningful gradients
    Call this after loss.backward() but before optimizer.step()
    """
    print("\n" + "="*80)
    print("GAMMA GRADIENT CHECK")
    print("="*80)
    
    for name, param in model.named_parameters():
        if 'gamma' in name and param.grad is not None:
            grad_magnitude = param.grad.abs().item()
            
            if grad_magnitude > 1e-3:
                status = "✅ STRONG gradient"
            elif grad_magnitude > 1e-5:
                status = "📉 WEAK gradient"
            else:
                status = "❌ VANISHING gradient"
            
            print(f"  {name:60s}")
            print(f"    Value: {param.item():+.6f}, Gradient: {param.grad.item():+.8f}  {status}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Example usage for testing
    import sys
    from pathlib import Path
    
    # Add source to path
    source_path = Path(__file__).parent.parent / "source"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    
    from model.vae_multi_input import MultiInputVAE
    
    print("Creating model...")
    model = MultiInputVAE(latent_dim=128, use_cross_attn=True)
    
    print("\nInitial gamma values (should all be 0):")
    log_gamma_values(model, epoch=0)
    
    # Simulate some training
    print("\nSimulating training (random update)...")
    for name, param in model.named_parameters():
        if 'gamma' in name:
            # Simulate gradient update
            param.data += torch.randn_like(param) * 0.1
    
    print("\nGamma values after simulated training:")
    log_gamma_values(model, epoch=10)
    
    print("\nAttention usage summary:")
    summary = get_attention_usage_summary(model)
    print(f"  Active: {summary['active']}/{summary['total']}")
    print(f"  Learning: {summary['learning']}/{summary['total']}")
    print(f"  Inactive: {summary['inactive']}/{summary['total']}")
