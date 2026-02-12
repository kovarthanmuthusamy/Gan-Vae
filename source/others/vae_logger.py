"""
Logging utilities for Variational Autoencoder training

Reference: src/logger.py
Adapted for VAE with multi-input, multi-output architecture
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List
import torch


class VAETrainingLogger:
    """Logger for tracking and visualizing VAE training metrics"""
    
    def __init__(self, log_dir: str, checkpoint_dir: Optional[str] = None):
        """
        Initialize the VAE logger
        
        Args:
            log_dir: Directory to save logs and metrics
            checkpoint_dir: Directory to save checkpoints (default: parent of log_dir/checkpoints)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = self.log_dir.parent / "checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        
        # Metric storage
        self.epochs = []
        self.total_loss = []
        self.recon_loss = []
        self.kl_loss = []
        self.heatmap_loss = []
        self.occupancy_loss = []
        self.impedance_loss = []
        
        # Initialize CSV file
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'total_loss', 'recon_loss', 'kl_loss',
                'heatmap_loss', 'occupancy_loss', 'impedance_loss'
            ])
    
    def log(self, epoch: int,
            total_loss: float,
            recon_loss: float,
            kl_loss: float,
            heatmap_loss: float,
            occupancy_loss: float,
            impedance_loss: float):
        """
        Log losses for a training epoch
        
        Args:
            epoch: Epoch number
            total_loss: Total VAE loss
            recon_loss: Reconstruction loss (weighted sum)
            kl_loss: KL divergence loss
            heatmap_loss: Heatmap reconstruction loss
            occupancy_loss: Occupancy reconstruction loss
            impedance_loss: Impedance reconstruction loss
        """
        self.epochs.append(epoch)
        self.total_loss.append(total_loss)
        self.recon_loss.append(recon_loss)
        self.kl_loss.append(kl_loss)
        self.heatmap_loss.append(heatmap_loss)
        self.occupancy_loss.append(occupancy_loss)
        self.impedance_loss.append(impedance_loss)
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, total_loss, recon_loss, kl_loss,
                heatmap_loss, occupancy_loss, impedance_loss
            ])
        
        # Console output
        print(f"Ep {epoch:3d} | Total: {total_loss:8.4f} | Recon: {recon_loss:8.4f} | KL: {kl_loss:8.4f} | "
              f"HM: {heatmap_loss:8.4f} | OC: {occupancy_loss:8.4f} | IMP: {impedance_loss:8.4f}")
    
    def log_dict(self, epoch: int, loss_dict: dict):
        """
        Log losses from a loss dictionary
        
        Args:
            epoch: Epoch number
            loss_dict: Dictionary with loss components from VAELoss
        """
        self.log(
            epoch=epoch,
            total_loss=loss_dict['total_loss'],
            recon_loss=loss_dict['recon_loss'],
            kl_loss=loss_dict['kl_loss'],
            heatmap_loss=loss_dict['heatmap_loss'],
            occupancy_loss=loss_dict['occupancy_loss'],
            impedance_loss=loss_dict['impedance_loss']
        )

    # ------------------------------------------------------------------
    # Console helpers for consistent messaging during training
    # ------------------------------------------------------------------

    def info(self, message: str):
        print(message)

    def log_environment(self, device, exp_path, log_path, checkpoint_path):
        self.info(f"Device: {device}")
        self.info(f"Experiment: {exp_path}")
        self.info(f"Logs: {log_path}")
        self.info(f"Checkpoints: {checkpoint_path}")

    def log_beta_schedule(self, start_epoch: int, end_epoch: int, max_beta: float):
        self.info(f"KL Annealing: β ramps 0 → {max_beta} (epochs {start_epoch}-{end_epoch})")

    def log_data_source(self, data_dir: str):
        self.info(f"\nLoading data from: {data_dir}")

    def log_dataset_overview(self, dataset_size, num_batches: int):
        self.info(f"\nDataset: {dataset_size} samples, {num_batches} batches")

    def log_epoch_progress(self, epoch: int, num_epochs: int, beta: float,
                           train_loss: float, val_loss: float):
        self.info(
            f"Epoch {epoch}/{num_epochs} | β={beta:.4f} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
        )

    def log_gamma_summary(self, active: int, total: int):
        self.info(f"\n✅ Training Complete! {active}/{total} attention layers active (|γ| > 0.1)")
    
    def save_checkpoint(self, epoch: int, model: torch.nn.Module, 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       checkpoint_dir: Optional[str] = None):
        """
        Save model checkpoint
        
        Args:
            epoch: Epoch number
            model: VAE model to save
            optimizer: Optional optimizer state
            checkpoint_dir: Directory to save checkpoint (default: self.checkpoint_dir)
        """
        try:
            if checkpoint_dir is None:
                ckpt_dir = self.checkpoint_dir
            else:
                ckpt_dir = Path(checkpoint_dir)
            
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            
            print(f"[DEBUG] Saving checkpoint to: {ckpt_path}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }
            
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(checkpoint, ckpt_path)
            print(f"✓ Checkpoint saved: {ckpt_path}")
        except Exception as e:
            print(f"❌ ERROR saving checkpoint at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
    
    def plot(self, save_path: Optional[str] = None):
        """
        Generate and save training visualization plots
        
        Args:
            save_path: Optional custom path to save plots
        """
        if not self.epochs:
            print("No data to plot")
            return
        
        epochs = np.array(self.epochs)
        total_loss = np.array(self.total_loss)
        recon_loss = np.array(self.recon_loss)
        kl_loss = np.array(self.kl_loss)
        heatmap_loss = np.array(self.heatmap_loss)
        occupancy_loss = np.array(self.occupancy_loss)
        impedance_loss = np.array(self.impedance_loss)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total and component losses
        axes[0, 0].plot(epochs, total_loss, 'b-', label='Total Loss', linewidth=2)
        axes[0, 0].plot(epochs, recon_loss, 'g-', label='Reconstruction Loss', linewidth=2)
        axes[0, 0].plot(epochs, kl_loss, 'r-', label='KL Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss vs Components')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Reconstruction loss components
        axes[0, 1].plot(epochs, heatmap_loss, 'orange', label='Heatmap Loss', linewidth=2)
        axes[0, 1].plot(epochs, occupancy_loss, 'purple', label='Occupancy Loss', linewidth=2)
        axes[0, 1].plot(epochs, impedance_loss, 'brown', label='Impedance Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Modality-specific Reconstruction Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # KL loss trend
        axes[1, 0].plot(epochs, kl_loss, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].set_title('KL Divergence Loss (Latent Space Regularization)')
        axes[1, 0].grid(alpha=0.3)
        
        # Loss ratio (recon vs KL)
        axes[1, 1].plot(epochs, recon_loss / (kl_loss + 1e-8), linewidth=2, color='cyan')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].set_title('Reconstruction to KL Loss Ratio')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = str(self.log_dir / "convergence.png")
        
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Plot saved: {save_path}")
    
    def plot_loss_components(self, save_path: Optional[str] = None):
        """
        Generate detailed loss component visualization
        
        Args:
            save_path: Optional custom path to save plots
        """
        if not self.epochs:
            print("No data to plot")
            return
        
        epochs = np.array(self.epochs)
        heatmap_loss = np.array(self.heatmap_loss)
        occupancy_loss = np.array(self.occupancy_loss)
        impedance_loss = np.array(self.impedance_loss)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(epochs, heatmap_loss, 'o-', linewidth=2, markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Heatmap Reconstruction Loss (64x64x2)')
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(epochs, occupancy_loss, 's-', linewidth=2, markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Occupancy Reconstruction Loss (7x8x1)')
        axes[1].grid(alpha=0.3)
        
        axes[2].plot(epochs, impedance_loss, '^-', linewidth=2, markersize=4)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Impedance Reconstruction Loss (231x1)')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.log_dir / "loss_components.png")
        
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Loss components plot saved: {save_path}")
    
    def get_statistics(self) -> dict:
        """
        Get training statistics summary
        
        Returns:
            Dictionary with min, max, mean, and final values for each loss
        """
        stats = {}
        
        for loss_name, loss_values in [
            ('total_loss', self.total_loss),
            ('recon_loss', self.recon_loss),
            ('kl_loss', self.kl_loss),
            ('heatmap_loss', self.heatmap_loss),
            ('occupancy_loss', self.occupancy_loss),
            ('impedance_loss', self.impedance_loss),
        ]:
            if loss_values:
                arr = np.array(loss_values)
                stats[loss_name] = {
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'mean': float(np.mean(arr)),
                    'final': float(arr[-1]),
                }
        
        return stats
    
    def print_statistics(self):
        """Print training statistics summary"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("TRAINING STATISTICS SUMMARY")
        print("="*80)
        
        for loss_name, values in stats.items():
            print(f"\n{loss_name}:")
            print(f"  Min:   {values['min']:.6f}")
            print(f"  Max:   {values['max']:.6f}")
            print(f"  Mean:  {values['mean']:.6f}")
            print(f"  Final: {values['final']:.6f}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '/home/ubuntu/gan')
    
    # Create logger for exp009
    logger = VAETrainingLogger(
        log_dir='experiments/exp009/logs',
        checkpoint_dir='experiments/exp009/checkpoints'
    )
    
    # Simulate training
    for epoch in range(1, 6):
        total_loss = 10.0 - epoch * 0.5 + np.random.randn() * 0.1
        recon_loss = 8.0 - epoch * 0.4
        kl_loss = 2.0 - epoch * 0.1
        heatmap_loss = 2.5 - epoch * 0.1
        occupancy_loss = 2.5 - epoch * 0.1
        impedance_loss = 3.0 - epoch * 0.1
        
        logger.log(epoch, total_loss, recon_loss, kl_loss, 
                  heatmap_loss, occupancy_loss, impedance_loss)
    
    # Generate plots
    logger.plot()
    logger.plot_loss_components()
    
    # Print statistics
    logger.print_statistics()
