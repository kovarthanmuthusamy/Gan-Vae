#!/usr/bin/env python3
"""
Compute per-modality latent space statistics from the trained model + dataset.
Saves stats to a JSON file that inference can use for proper sampling.
"""
import sys, json, torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.model.vae_multi_input_simple import MultiInputVAE
from source.others.dataloader import create_data_loaders


def compute_latent_stats(checkpoint_path, data_root='datasets/data_norm', device='cuda'):
    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MultiInputVAE(latent_dim=160).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load full training data
    train_loader, val_loader = create_data_loaders(
        data_root, batch_size=256, num_workers=4,
        normalize=False, train_split=0.9, seed=42
    )

    # Collect all mu vectors per modality
    all_mu = {'heatmap': [], 'impedance': [], 'maxvalue': [], 'shared': []}
    all_std = {'heatmap': [], 'impedance': [], 'maxvalue': [], 'shared': []}

    print(f"Computing latent stats over {len(train_loader)} batches...")
    with torch.no_grad():
        for batch in train_loader:
            heatmap = batch['heatmap_norm'].to(device)
            occupancy = batch['occupancy'].to(device)
            impedance = batch['impedance'].to(device)
            maxvalue = batch['max_impedance_std'].to(device)
            if maxvalue.dim() == 1:
                maxvalue = maxvalue.unsqueeze(1)

            _, _, _, _, mu_gauss, logvar_gauss, _, mod_stats = model(
                heatmap, occupancy, impedance, maxvalue
            )

            # Per-modality private stats
            for name in ['heatmap', 'impedance', 'maxvalue']:
                mu_mod, logvar_mod = mod_stats[name]
                all_mu[name].append(mu_mod.cpu())
                all_std[name].append(torch.exp(0.5 * logvar_mod).cpu())

            # Shared latent (from PoE) — last shared_dim dims of mu_gauss
            shared_mu = mu_gauss[:, -model.shared_dim:]
            shared_logvar = logvar_gauss[:, -model.shared_dim:]
            all_mu['shared'].append(shared_mu.cpu())
            all_std['shared'].append(torch.exp(0.5 * shared_logvar).cpu())

    # Concatenate and compute statistics
    stats = {}
    for name in all_mu:
        mu_all = torch.cat(all_mu[name], dim=0)  # (N, dim)
        std_all = torch.cat(all_std[name], dim=0)

        # Per-dimension statistics of the mean vectors
        per_dim_mu_mean = mu_all.mean(dim=0)  # (dim,)
        per_dim_mu_std = mu_all.std(dim=0)    # (dim,)

        # Aggregate posterior std: sqrt(var(mu) + E[sigma^2])
        agg_std = torch.sqrt(mu_all.var(dim=0) + (std_all ** 2).mean(dim=0))

        stats[name] = {
            'mu_mean': per_dim_mu_mean.mean().item(),
            'mu_std': per_dim_mu_std.mean().item(),
            'mu_mean_per_dim': per_dim_mu_mean.tolist(),
            'mu_std_per_dim': per_dim_mu_std.tolist(),
            'sigma_mean': std_all.mean().item(),
            'agg_std': agg_std.mean().item(),
            'n_samples': mu_all.shape[0],
            'n_dims': mu_all.shape[1],
        }

        print(f"\n{name} (dim={mu_all.shape[1]}):")
        print(f"  mu:  mean={stats[name]['mu_mean']:.4f}, std={stats[name]['mu_std']:.4f}")
        print(f"  sigma: mean={stats[name]['sigma_mean']:.4f}")
        print(f"  aggregate posterior std: {stats[name]['agg_std']:.4f}")

    return stats


if __name__ == '__main__':
    ckpt_path = 'experiments/exp018/checkpoints/checkpoint_epoch_200.pt'
    stats = compute_latent_stats(ckpt_path)

    out_path = Path(ckpt_path).parent.parent / 'metrics' / 'latent_stats.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved latent stats to: {out_path}")
