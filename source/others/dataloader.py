"""
Data loader for Multi-Input VAE training

Loads three modalities from disk:
1. Heatmap: 1x64x64 (log(1+x) z-score normalized, background=sentinel)
2. Occupancy Vector: 52
3. Impedance Vector: 231 (log z-score normalized)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict
import json


class VAEDataset(Dataset):
    """PyTorch Dataset for VAE with three modalities"""
    
    def __init__(self, 
                 data_dir: str = "source/data_norm",
                 normalize: bool = False,
                 stats_path: Optional[str] = None):
        """
        Initialize the dataset
        
        Args:
            data_dir: Path to dataset directory containing heatmap/, Imp/, Occ_map/ subdirs
            normalize: Whether to normalize using stats
            stats_path: Path to normalization stats JSON file
        """
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.stats = None
        
        # Paths to modality directories
        self.heatmap_dir = self.data_dir / "heatmap"
        self.impedance_dir = self.data_dir / "Imp"
        self.occupancy_dir = self.data_dir / "Occ_map"
        
        # Validate directories exist
        for dir_path in [self.heatmap_dir, self.impedance_dir, self.occupancy_dir]:
            if not dir_path.exists():
                raise ValueError(f"Directory not found: {dir_path}")
        
        # Load file list from heatmap directory (all should have same count)
        self.heatmap_files = sorted(list(self.heatmap_dir.glob("*.npy")))
        if not self.heatmap_files:
            raise ValueError(f"No .npy files found in {self.heatmap_dir}")
        
        self.n_samples = len(self.heatmap_files)
        print(f"Loaded {self.n_samples} samples from {data_dir}")
        
        # Load normalization stats if normalizing
        if self.normalize:
            if stats_path is None:
                stats_path = str(self.data_dir / "normalization_stats.json")
            
            if Path(stats_path).exists():
                with open(stats_path, 'r') as f:
                    self.stats = json.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                print(f"Warning: stats file not found at {stats_path}, skipping normalization")
                self.normalize = False
    
    def __len__(self) -> int:
        """Return number of samples"""
        return self.n_samples
    
    def _get_filename(self, idx: int) -> str:
        """Get the base filename for a sample"""
        return self.heatmap_files[idx].stem
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        """
        Get a sample
        
        Returns:
            Dictionary with 'heatmap_norm', 'occupancy', 'impedance' tensors and 'filename' string
        """
        filename = self._get_filename(idx)
        
        # Load heatmap (1, 64, 64) - already normalized (log(1+x) z-score)
        heatmap_norm = np.load(self.heatmap_dir / f"{filename}.npy")  # (1, 64, 64)
        
        # Load impedance (231,)
        impedance = np.load(self.impedance_dir / f"{filename}.npy")  # (231,)
        
        # Load occupancy vector (52,)
        occupancy = np.load(self.occupancy_dir / f"{filename}.npy").flatten()  # (52,)
        
        # Convert to tensors
        heatmap_norm = torch.from_numpy(heatmap_norm).float()  # (1, 64, 64)
        impedance = torch.from_numpy(impedance).float().view(-1)  # (231,)
        occupancy = torch.from_numpy(occupancy).float()  # (52,)
        occupancy = torch.clamp(occupancy, 0.0, 1.0)
        
        return {
            'heatmap_norm': heatmap_norm,
            'occupancy': occupancy,
            'impedance': impedance,
            'filename': filename
        }


def create_data_loaders(
    data_dir: str = "src/data_norm",
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = False,
    stats_path: Optional[str] = None,
    train_split: float = 0.8,
    seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        normalize: Whether to normalize data
        stats_path: Path to normalization stats
        train_split: Train/validation split ratio
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = VAEDataset(
        data_dir=data_dir,
        normalize=normalize,
        stats_path=stats_path
    )
    
    # Split dataset
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    n_val = n_samples - n_train
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )
    
    print(f"Train samples: {n_train}, Validation samples: {n_val}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def collate_fn(batch: list) -> Dict[str, torch.Tensor | list[str]]:
    """
    Custom collate function for batching
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Dictionary with stacked tensors and filenames list
    """
    heatmaps = torch.stack([item['heatmap_norm'] for item in batch])
    occupancies = torch.stack([item['occupancy'] for item in batch])
    impedances = torch.stack([item['impedance'] for item in batch])
    filenames = [item['filename'] for item in batch]
    
    return {
        'heatmap_norm': heatmaps,
        'occupancy': occupancies,
        'impedance': impedances,
        'filenames': filenames
    }


if __name__ == "__main__":
    # Remove unused imports
    import sys
    from pathlib import Path
    
    # Add workspace root to path
    workspace_root = Path(__file__).parent.parent.parent
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    
    # Test the dataloader
    print("Testing VAE DataLoader...")
    
    # Create loaders
    train_loader, val_loader = create_data_loaders(
        data_dir="datasets/data_norm",
        batch_size=4,
        num_workers=0,
        normalize=False,
        train_split=0.8
    )
    
    # Get a batch
    print("\nGetting a sample batch...")
    for batch in train_loader:
        print(f"Heatmap shape: {batch['heatmap_norm'].shape}")
        print(f"Occupancy shape: {batch['occupancy'].shape}")
        print(f"Impedance shape: {batch['impedance'].shape}")
        print(f"Filenames: {batch['filenames']}")
        print(f"Heatmap range: [{batch['heatmap_norm'].min():.3f}, {batch['heatmap_norm'].max():.3f}]")
        break
    
    print("\nDataLoader test completed successfully!")
