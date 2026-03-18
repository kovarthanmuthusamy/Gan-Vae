"""
Data loader for Multi-Input VAE training

Loads three modalities from disk:
1. Heatmap: 64x64x2
2. Occupancy Vector: 52
3. Impedance Vector: 231x1
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict
import json


class VAEDataset(Dataset):
    """PyTorch Dataset for VAE with three modalities + max_impedance"""
    
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
        self.max_value_dir = self.data_dir / "Max_value"  # NEW: max value directory
        
        # Validate directories exist
        for dir_path in [self.heatmap_dir, self.impedance_dir, self.occupancy_dir]:
            if not dir_path.exists():
                raise ValueError(f"Directory not found: {dir_path}")
        
        # Check if Max_value directory exists (optional for backward compatibility)
        self.has_max_values = self.max_value_dir.exists()
        if not self.has_max_values:
            print(f"Warning: Max_value directory not found at {self.max_value_dir}")
            print("         Will compute max_impedance from normalized heatmap (less accurate)")
        
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
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with 'heatmap_norm', 'max_impedance_std', 'occupancy', 'impedance' tensors and 'filename' string
        """
        filename = self._get_filename(idx)
        
        # Load heatmap (2, 64, 64) - already NORMALIZED in (C, H, W) format
        heatmap_norm = np.load(self.heatmap_dir / f"{filename}.npy")  # (2, 64, 64)
        
        # Load impedance (231,)
        impedance = np.load(self.impedance_dir / f"{filename}.npy")  # (231,)
        
        # Load occupancy vector (52,)
        occupancy = np.load(self.occupancy_dir / f"{filename}.npy").flatten()  # (52,)
        
        # Apply normalization if available (for impedance only)
        if self.normalize and self.stats:
            impedance = self._normalize(impedance, 'impedance')
        
        # 🎯 PHYSICALLY-AWARE: Load max_impedance from stored file (already Z-score normalized)
        if self.has_max_values:
            # Load the stored normalized max value (already normalized in Normalization.py)
            max_impedance_normalized = np.load(self.max_value_dir / f"{filename}.npy")
            max_impedance_normalized = float(max_impedance_normalized.item())  # Convert to scalar
        else:
            # Fallback: approximate from normalized heatmap (not ideal but backward compatible)
            max_impedance_normalized = np.abs(heatmap_norm).max()
            if max_impedance_normalized < 1e-8:
                max_impedance_normalized = 0.0  # Use 0 as default normalized value
        
        # Convert to tensors
        heatmap_norm = torch.from_numpy(heatmap_norm).float()  # (2, 64, 64)
        max_impedance_std = torch.tensor([max_impedance_normalized], dtype=torch.float32)  # (1,) - already normalized
        impedance = torch.from_numpy(impedance).float().view(-1)  # (231,)
        occupancy = torch.from_numpy(occupancy).float()  # (52,)
        occupancy = torch.clamp(occupancy, 0.0, 1.0)
        
        return {
            'heatmap_norm': heatmap_norm,
            'max_impedance_std': max_impedance_std,
            'occupancy': occupancy,
            'impedance': impedance,
            'filename': filename
        }
    
    def _normalize(self, data: np.ndarray, modality: str) -> np.ndarray:
        """
        Normalize data using loaded statistics
        
        Args:
            data: Data array
            modality: 'heatmap', 'impedance', or 'occupancy'
        
        Returns:
            Normalized data
        """
        if self.stats is None or modality not in self.stats:
            return data
        
        stats = self.stats[modality]
        mean = np.array(stats.get('mean', 0))
        std = np.array(stats.get('std', 1))
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        return (data - mean) / std


def compute_max_impedance_statistics(
    data_dir: str = "src/data_norm",
    num_samples: Optional[int] = None
) -> Tuple[float, float]:
    """
    [DEPRECATED] Compute mean and std of max_impedance values across dataset.
    
    ⚠️ This function is no longer needed if you use the updated Normalization.py script,
    which now computes and applies Z-score normalization to max_values automatically.
    The normalized max_values are saved directly in Max_value/ folder.
    
    This function is kept for backward compatibility with old datasets.
    
    Args:
        data_dir: Path to dataset directory
        num_samples: Optional limit on number of samples to use (for speed)
    
    Returns:
        Tuple of (mean, std) for max_impedance
    """
    data_path = Path(data_dir)
    max_value_dir = data_path / "Max_value"
    heatmap_dir = data_path / "heatmap"
    
    # Check if we have stored max values (more accurate)
    if max_value_dir.exists():
        print(f"Using stored max values from {max_value_dir}")
        max_value_files = sorted(list(max_value_dir.glob("*.npy")))
        
        if not max_value_files:
            print(f"Warning: Max_value directory exists but is empty, falling back to heatmap computation")
            use_stored = False
        else:
            use_stored = True
    else:
        print(f"Max_value directory not found, computing from normalized heatmaps")
        use_stored = False
        max_value_files = []
    
    if not use_stored:
        # Fallback: compute from heatmaps
        heatmap_files = sorted(list(heatmap_dir.glob("*.npy")))
        if not heatmap_files:
            raise ValueError(f"No heatmap files found in {heatmap_dir}")
        files_to_process = heatmap_files
    else:
        files_to_process = max_value_files
    
    # Optionally sample subset for speed
    if num_samples is not None and len(files_to_process) > num_samples:
        indices = np.random.choice(len(files_to_process), num_samples, replace=False)
        files_to_process = [files_to_process[i] for i in sorted(indices)]
    
    print(f"Computing max_impedance statistics from {len(files_to_process)} samples...")
    
    max_impedances = []
    for file_path in files_to_process:
        if use_stored:
            # Load stored max value (exact)
            max_imp_array = np.load(file_path)
            max_imp = float(max_imp_array.item())
        else:
            # Compute from heatmap (approximate)
            heatmap = np.load(file_path)
            max_imp = np.abs(heatmap).max()
            if max_imp < 1e-8:
                max_imp = 1.0
        max_impedances.append(max_imp)
    
    max_impedances = np.array(max_impedances)
    mean = float(np.mean(max_impedances))
    std = float(np.std(max_impedances))
    
    print(f"📊 Max Impedance Statistics:")
    print(f"   Mean: {mean:.6f}")
    print(f"   Std:  {std:.6f}")
    print(f"   Min:  {max_impedances.min():.6f}")
    print(f"   Max:  {max_impedances.max():.6f}")
    
    return mean, std


def create_data_loaders(
    data_dir: str = "src/data_norm",
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = False,
    stats_path: Optional[str] = None,
    train_split: float = 0.8,
    seed: int = 42) -> Tuple[DataLoader, DataLoader]:
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
    max_impedances = torch.stack([item['max_impedance_std'] for item in batch])
    occupancies = torch.stack([item['occupancy'] for item in batch])
    impedances = torch.stack([item['impedance'] for item in batch])
    filenames = [item['filename'] for item in batch]
    
    return {
        'heatmap_norm': heatmaps,
        'max_impedance_std': max_impedances,
        'occupancy': occupancies,
        'impedance': impedances,
        'filenames': filenames
    }


if __name__ == "__main__":
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
        data_dir="src/data_norm",
        batch_size=4,
        num_workers=0,
        normalize=False,
        train_split=0.8
    )
    
    # Compute max_impedance statistics first
    print("\nComputing max_impedance statistics...")
    mean, std = compute_max_impedance_statistics(data_dir="src/data_norm", num_samples=100)
    
    # Get a batch
    print("\nGetting a sample batch...")
    for batch in train_loader:
        print(f"Heatmap shape: {batch['heatmap_norm'].shape}")
        print(f"Max Impedance (std) shape: {batch['max_impedance_std'].shape}")
        print(f"Occupancy shape: {batch['occupancy'].shape}")
        print(f"Impedance shape: {batch['impedance'].shape}")
        print(f"Filenames: {batch['filenames']}")
        print(f"Max Impedance (std) range: [{batch['max_impedance_std'].min():.3f}, {batch['max_impedance_std'].max():.3f}]")
        break
    
    print("\nDataLoader test completed successfully!")
