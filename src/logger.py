import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        
        self.epochs = []
        self.loss_g = []
        self.loss_d = []
        self.loss_adv = []
        self.loss_fm = []  # Feature matching loss
        self.loss_gp = []  # Gradient penalty
        self.grad_norm = []
        
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'loss_g', 'loss_d', 'loss_adv', 'loss_fm', 'loss_gp', 'grad_norm'])
    
    def log(self, epoch, loss_g, loss_d, loss_adv=0.0, loss_fm=0.0, loss_gp=0.0, grad_norm=0.0):
        self.epochs.append(epoch)
        self.loss_g.append(loss_g)
        self.loss_d.append(loss_d)
        self.loss_adv.append(loss_adv)
        self.loss_fm.append(loss_fm)
        self.loss_gp.append(loss_gp)
        self.grad_norm.append(grad_norm)
        
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, loss_g, loss_d, loss_adv, loss_fm, loss_gp, grad_norm])
        
        print(f"Ep {epoch} | Loss_G: {loss_g:.4f} | Loss_D: {loss_d:.4f} | Adv: {loss_adv:.4f} | FM: {loss_fm:.4f} | GP: {loss_gp:.4f} | GradNorm: {grad_norm:.4f}")
    
    def save_checkpoint(self, epoch, G, D):
        """Save checkpoint with epoch naming (e.g., epoch_10.pt, epoch_20.pt)"""
        import torch
        ckpt_path = self.log_dir.parent / "checkpoints" / f"epoch_{epoch}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict()}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
    
    def plot(self):
        if not self.epochs:
            return
        
        epochs = np.array(self.epochs)
        loss_g = np.array(self.loss_g)
        loss_d = np.array(self.loss_d)
        loss_adv = np.array(self.loss_adv)
        loss_fm = np.array(self.loss_fm)
        loss_gp = np.array(self.loss_gp)
        grad_norm = np.array(self.grad_norm)
        
        # Create subplots for different loss components
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Main losses (G and D)
        axes[0, 0].plot(epochs, loss_g, 'b-', label='Generator Loss', linewidth=2)
        axes[0, 0].plot(epochs, loss_d, 'r-', label='Discriminator Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator vs Discriminator Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Generator loss components
        axes[0, 1].plot(epochs, loss_adv, 'g-', label='Adversarial Loss', linewidth=2)
        axes[0, 1].plot(epochs, loss_fm, 'orange', label='Feature Matching Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Generator Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Feature matching loss
        axes[1, 0].plot(epochs, loss_fm, 'orange', label='Feature Matching Loss', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Feature Matching Loss (MAE on fusion_conv2)')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Gradient penalty
        axes[1, 1].plot(epochs, loss_gp, 'cyan', label='Gradient Penalty', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Gradient Penalty')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / "convergence.png", dpi=300)
        plt.close()
        
        print(f"Plot saved to {self.log_dir / 'convergence.png'}")
