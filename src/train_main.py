import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import random

from models.model_v1 import Generator, Critic
from losses.loss_fn import (
    critic_loss, generator_loss,
    binarize_occupancy, binarize_impedance,
    feature_matching_loss
)
from logger import TrainingLogger

CONFIG = {
    "batch_size": 32,
    "num_epochs": 150,
    "lr_g": 0.00005,
    "lr_d": 0.00005,
    "beta1": 0.0,
    "beta2": 0.9,
    "latent_dim": 200,
    "shared_dim": 512,
    "lambda_gp": 100.0,
    "critic_iter": 4,
    "max_samples": None,
    "device": "cuda:1",
    "checkpoint_dir": "/home/ubuntu/gan/experiments/exp008/checkpoints",
    "logs_dir": "/home/ubuntu/gan/experiments/exp008/logs",
    "mask_path": "/home/ubuntu/gan/configs/binary_mask.npy",
    # Lambda weights for losses
    "lambda_adv": 1.0,
    "lambda_fm": 10.0,  # Feature matching loss weight
    # Resume training from latest checkpoint
    "resume": True,
    "epsilon_drift": 0.001,
}

class HeatmapImpedanceDataset(Dataset):
    def __init__(self, heatmap_dir, impedance_dir, occupancy_dir, max_samples=None, random_seed=42):
        self.heatmap_dir = Path(heatmap_dir)
        self.impedance_dir = Path(impedance_dir)
        self.occupancy_dir = Path(occupancy_dir)
        
        heatmap_files = sorted([f for f in self.heatmap_dir.glob("*.npy")])
        impedance_files = sorted([f for f in self.impedance_dir.glob("*.npy")])
        occupancy_files = sorted([f for f in self.occupancy_dir.glob("*.npy")])
        
        if not heatmap_files or not impedance_files or not occupancy_files:
            raise ValueError("No .npy files found in heatmap, impedance, or occupancy directories")
        
        heatmap_names = {f.stem: f for f in heatmap_files}
        impedance_names = {f.stem: f for f in impedance_files}
        occupancy_names = {f.stem: f for f in occupancy_files}
        
        self.file_pairs = [(heatmap_names[name], impedance_names[name], occupancy_names[name]) 
                          for name in heatmap_names if name in impedance_names and name in occupancy_names]
        self.file_pairs = sorted(self.file_pairs)
        
        # Randomly select max_samples instead of taking first max_samples
        if max_samples and max_samples < len(self.file_pairs):
            random.seed(random_seed)
            self.file_pairs = random.sample(self.file_pairs, max_samples)
            self.file_pairs = sorted(self.file_pairs)  # Keep sorted for reproducibility
        
        print(f"Loaded {len(self.file_pairs)} data triplets (heatmap, impedance, occupancy)")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        heatmap_path, impedance_path, occupancy_path = self.file_pairs[idx]
        heatmap = torch.tensor(np.load(heatmap_path), dtype=torch.float32)  # (2, 64, 64) - already C,H,W
        impedance = torch.tensor(np.load(impedance_path), dtype=torch.float32).flatten()  # (231,)
        occupancy = torch.tensor(np.load(occupancy_path), dtype=torch.float32)  # (7, 8) or (7, 8, 1)
        
        # Ensure occupancy has shape (1, 7, 8)
        if occupancy.ndim == 2:
            occupancy = occupancy.unsqueeze(0)
        elif occupancy.ndim == 3 and occupancy.shape[-1] == 1:
            occupancy = occupancy.permute(2, 0, 1)
        
        return heatmap, impedance, occupancy


def get_dataloader(heatmap_dir, impedance_dir, occupancy_dir, batch_size=32, num_workers=4, max_samples=None, split='train', val_split=0.1):
    dataset = HeatmapImpedanceDataset(heatmap_dir, impedance_dir, occupancy_dir, max_samples)
    
    # Split into train and validation
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    if split == 'train':
        indices = list(range(train_size))
    elif split == 'val':
        indices = list(range(train_size, dataset_size))
    else:
        raise ValueError("split must be 'train' or 'val'")
    
    from torch.utils.data import Subset
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers, pin_memory=True)


def train_epoch(G, D, dataloader, opt_G, opt_D, config, mask):
    device = config["device"]
    latent_dim = config["latent_dim"]

    def _grad_norm(model):
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                n = p.grad.detach().norm(2).item()
                total += n * n
        return total ** 0.5
    
    total_loss_g = 0.0
    total_loss_d = 0.0
    total_grad_norm = 0.0
    total_grad_norm_g = 0.0
    total_gp = 0.0
    total_d_real = 0.0
    total_d_fake = 0.0
    total_fm = 0.0  # Feature matching loss
    total_adv = 0.0
    num_batches = 0
    
    # Adaptive balancing alpha (persistence across batches)
    # Initialize with 1.0 or based on first batch (will be updated)
    current_alpha = 1.0
    balancing_interval = 10  # Update alpha every N steps

    for real_h, real_i, real_o in dataloader:
        real_h, real_i, real_o = real_h.to(device), real_i.to(device), real_o.to(device)
        batch_size = real_h.size(0)

        # Apply mask to real heatmap
        real_h_masked = real_h * mask

        # Train Discriminator
        for _ in range(config["critic_iter"]):
            opt_D.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_h, fake_o, fake_i = G(z)

            # Apply mask to fake heatmap
            fake_h_masked = fake_h * mask

            loss_d, grad_norm_mean, gp_val, d_real_mean, d_fake_mean = critic_loss(
                D,
                real_h_masked, real_o, real_i,
                fake_h_masked, fake_o, fake_i,
                config["lambda_gp"], device, mask=mask, epsilon_drift=config.get("epsilon_drift", 1e-3)
            )
            loss_d.backward()
            d_grad = _grad_norm(D)
            opt_D.step()
            total_loss_d += loss_d.item()
            total_grad_norm += d_grad
            total_gp += gp_val
            total_d_real += d_real_mean
            total_d_fake += d_fake_mean
            num_batches += 1

        # Train Generator
        opt_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_h, fake_o, fake_i = G(z)
        fake_h_masked = fake_h * mask

        # Adversarial loss
        loss_g_adv = generator_loss(D, fake_h_masked, fake_o, fake_i) * config["lambda_adv"]

        # Feature matching loss (compares intermediate feature maps from critic's fusion_conv2)
        loss_fm = feature_matching_loss(
            D,
            real_h_masked, real_o, real_i,
            fake_h_masked, fake_o, fake_i
        ) * config["lambda_fm"]

        # Gradient balancing (Adaptive Alpha)
        # Update alpha every N steps
        if num_batches % balancing_interval == 0:
            # Measure gradients
            g_adv = torch.autograd.grad(loss_g_adv, G.parameters(), retain_graph=True)
            g_fm = torch.autograd.grad(loss_fm, G.parameters(), retain_graph=True)
            
            # Compute norms
            flat_adv = torch.cat([p.flatten() for p in g_adv])
            flat_fm = torch.cat([p.flatten() for p in g_fm])
            norm_adv = torch.norm(flat_adv)
            norm_fm = torch.norm(flat_fm)
            
            current_alpha = (norm_adv / (norm_fm + 1e-8)).detach()
            # print(f"Updated Alpha: {current_alpha.item():.4f} (Adv Norm: {norm_adv.item():.4f}, FM Norm: {norm_fm.item():.4f})")

        # Total generator loss with balanced feature matching
        loss_g = loss_g_adv + current_alpha * loss_fm
        loss_g.backward()
        g_grad = _grad_norm(G)
        opt_G.step()
        
        total_adv += loss_g_adv.item()
        total_fm += loss_fm.item()
        total_loss_g += loss_g.item()
        total_grad_norm_g += g_grad
    
    avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
    avg_adv = total_adv / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_fm = total_fm / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_gp = total_gp / num_batches if num_batches > 0 else 0.0
    avg_d_real = total_d_real / num_batches if num_batches > 0 else 0.0
    avg_d_fake = total_d_fake / num_batches if num_batches > 0 else 0.0
    avg_grad_norm_g = total_grad_norm_g / num_batches if num_batches > 0 else 0.0
    
    print(f"[Epoch] adv:{avg_adv:.4f} fm:{avg_fm:.4f} gp:{avg_gp:.4f} d_real:{avg_d_real:.4f} d_fake:{avg_d_fake:.4f} g_grad:{avg_grad_norm_g:.4f}")
    return (
        total_loss_g / len(dataloader),
        total_loss_d / (len(dataloader) * config["critic_iter"]),
        avg_grad_norm,
        avg_adv,
        avg_fm,
        avg_gp
    )


def validate(G, D, dataloader, config, mask):
    """Evaluate on validation set (no gradient penalty, faster validation)"""
    device = config["device"]
    latent_dim = config["latent_dim"]
    
    total_loss_g = 0.0
    total_loss_d = 0.0
    
    G.eval()
    D.eval()
    with torch.no_grad():
        for real_h, real_i, real_o in dataloader:
            real_h, real_i, real_o = real_h.to(device), real_i.to(device), real_o.to(device)
            batch_size = real_h.size(0)
            
            real_h_masked = real_h * mask
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_h, fake_o, fake_i = G(z)
            fake_h_masked = fake_h * mask
            
            # Compute losses without gradient penalty (no constraint needed in validation)
            real_score = D(real_h_masked, real_o, real_i)
            fake_score = D(fake_h_masked, fake_o, fake_i)
            
            # Wasserstein distance without GP
            loss_d = fake_score.mean() - real_score.mean()
            loss_g = -fake_score.mean()
            
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
    
    G.train()
    D.train()
    return total_loss_g / len(dataloader), total_loss_d / len(dataloader)


def generate_samples(G, num_samples, config, binarize=False, device="cuda"):
    """
    Generate samples from the generator.
    
    Args:
        G: Generator model
        num_samples: Number of samples to generate
        config: Configuration dict
        binarize: If True, binarize occupancy and impedance outputs during inference
        device: Device to use
    
    Returns:
        Tuple of (heatmaps, occupancy, impedance)
    """
    G.eval()
    device = torch.device(device)
    latent_dim = config["latent_dim"]
    
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        fake_h, fake_o, fake_i = G(z)
        
        # Optionally binarize occupancy and impedance during inference
        if binarize:
            fake_o = binarize_occupancy(fake_o, threshold=0.5)
            fake_i = binarize_impedance(fake_i, threshold=0.5)
    
    G.train()
    return fake_h.cpu(), fake_o.cpu(), fake_i.cpu()


def main():
    device = torch.device(CONFIG["device"])
    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["logs_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Load mask
    mask_data = np.load(CONFIG["mask_path"]).astype(np.float32)
    mask = torch.tensor(mask_data, dtype=torch.float32).to(device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # Add channel dimension (1, H, W)
    # Replicate mask to 2 channels (impedance, mask)
    mask = mask.repeat(2, 1, 1) if mask.size(0) == 1 else mask
    
    logger = TrainingLogger(CONFIG["logs_dir"])
    
    G = Generator(CONFIG["latent_dim"], CONFIG["shared_dim"]).to(device)
    D = Critic().to(device)
    
    opt_G = optim.Adam(G.parameters(), lr=CONFIG["lr_g"], betas=(CONFIG["beta1"], CONFIG["beta2"]))
    opt_D = optim.Adam(D.parameters(), lr=CONFIG["lr_d"], betas=(CONFIG["beta1"], CONFIG["beta2"]))

    # Optional resume from latest checkpoint in checkpoint_dir
    start_epoch = 0
    if CONFIG.get("resume", False):
        ckpt_dir = Path(CONFIG["checkpoint_dir"])
        ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
        if ckpts:
            latest_ckpt = ckpts[-1]
            ckpt = torch.load(latest_ckpt, map_location=device)
            G.load_state_dict(ckpt["G"])
            D.load_state_dict(ckpt["D"])
            start_epoch = ckpt.get("epoch", 0)
            print(f"Resumed from {latest_ckpt} at epoch {start_epoch}")
        else:
            print("Resume enabled but no checkpoint found; starting from scratch.")
    
    # Load train and validation dataloaders
    repo_root = Path(__file__).resolve().parents[1]
    heatmap_path = repo_root / "src" / "data_norm" / "heatmap"
    impedance_path = repo_root / "src" / "data_norm" / "Imp"
    occupancy_path = repo_root / "src" / "data_norm" / "Occ_map"
    
    train_dataloader = get_dataloader(str(heatmap_path), str(impedance_path), str(occupancy_path),
                                      CONFIG["batch_size"], 4, CONFIG["max_samples"], split='train')
    val_dataloader = get_dataloader(str(heatmap_path), str(impedance_path), str(occupancy_path),
                                    CONFIG["batch_size"], 4, CONFIG["max_samples"], split='val')
    
    print(f"Training on {device}...")
    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        # Training phase
        loss_g, loss_d, grad_norm, loss_adv, loss_fm, avg_gp = train_epoch(G, D, train_dataloader, opt_G, opt_D, CONFIG, mask)
        
        # Log all losses to CSV and internal arrays for plotting
        logger.log(epoch + 1, loss_g, loss_d, loss_adv, loss_fm, avg_gp, grad_norm)
        
        # Print training info (every epoch)
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']} | Loss_G: {loss_g:.4f} | Loss_D: {loss_d:.4f} | Grad_Norm: {grad_norm:.4f}")
        
        # Validation phase (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            val_loss_g, val_loss_d = validate(G, D, val_dataloader, CONFIG, mask)
            print(f"  Val - Loss_G: {val_loss_g:.4f}, Loss_D: {val_loss_d:.4f}")
        
        if (epoch + 1) % 10 == 0:
            logger.save_checkpoint(epoch + 1, G, D)
    
    logger.plot()
    print("Done!")


if __name__ == "__main__":
    main()
