"""Helper module for occupancy vector operations. Import only — not for direct execution."""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DECAP_VECTOR_LENGTH = 52


def create_occupancy_vector(decap_vector):
    """
    Create a normalised float32 occupancy vector from a raw decap vector.

    Args:
        decap_vector: array-like of length 52 (capacitor presence values)

    Returns:
        np.ndarray of shape (52,), dtype float32, values in {0.0, 1.0}
    """
    vec = np.asarray(decap_vector, dtype=np.float32).flatten()
    if len(vec) != DECAP_VECTOR_LENGTH:
        raise ValueError(
            f"Expected decap vector of length {DECAP_VECTOR_LENGTH}, got {len(vec)}"
        )
    return (vec > 0.5).astype(np.float32)


labels_v1 = [
    "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
    "C11","C12","C13","C14","C15","C16","C17","C18","C19","C20",
    "C21","C22","C23","C24","C25","C26","C27","C28","C29","C30",
    "C31","C32","C33","C34","C35","C36","C37","C38","C39","C40",
    "C41","C42","C43","C44","C45","C46","C47","C48","C49","C50",
    "C51","C52"
]


def visualize_occupancy_vector(vector_or_path):
    """
    Return the active capacitor labels (value > 0.5) from an occupancy vector.

    Args:
        vector_or_path: 1D numpy array (52,) or path to .npy file

    Returns:
        list of str — labels from labels_v1 where the vector value is 1
    """
    if isinstance(vector_or_path, (str, Path)):
        vector = np.load(vector_or_path).flatten()
    else:
        vector = np.asarray(vector_or_path).flatten()

    return [
        labels_v1[i]
        for i in range(min(len(vector), len(labels_v1)))
        if vector[i] > 0.5
    ]


def read_decap_csv(csv_path, max_samples=None, verbose=True):
    """
    Read decap vector CSV file.
    
    Args:
        csv_path: Path to CSV file containing decap vectors (one vector per row)
        max_samples: Maximum number of samples to process (None for all)
        verbose: Print progress information
    
    Returns:
        numpy array of shape (N, 52) where N is number of samples
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if verbose:
        print(f"Reading CSV file: {csv_path}")
    
    try:
        # Read CSV - assuming no header, all numeric data
        df = pd.read_csv(csv_path, header=None)
        
        # Extract numeric columns only
        decap_vectors = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        
        # Validate shape
        if decap_vectors.shape[1] != DECAP_VECTOR_LENGTH:
            print(f"Warning: Expected {DECAP_VECTOR_LENGTH} columns, got {decap_vectors.shape[1]}")
            print(f"         Will use first {min(decap_vectors.shape[1], DECAP_VECTOR_LENGTH)} values")
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(decap_vectors):
            decap_vectors = decap_vectors[:max_samples]
            if verbose:
                print(f"Limited to first {max_samples} samples")
        
        if verbose:
            print(f"Loaded {len(decap_vectors)} decap vectors")
            print(f"Decap vector shape: {decap_vectors.shape}")
            print(f"Value range: [{decap_vectors.min():.4f}, {decap_vectors.max():.4f}]")
            print(f"Non-zero ratio: {(decap_vectors > 0.5).sum() / decap_vectors.size:.2%}")
        
        return decap_vectors
        
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")


def convert_csv_to_occupancy(csv_path, output_dir, max_samples=None, 
                             start_index=1, verbose=True):
    """
    Convert CSV rows to occupancy grid samples.
    
    Args:
        csv_path: Path to CSV file containing decap vectors
        output_dir: Directory to save occupancy grid .npy files
        max_samples: Maximum number of samples to process (None for all)
        start_index: Starting index for sample numbering (default: 1)
        verbose: Print progress information
    
    Returns:
        Number of samples successfully created
    """
    # Read decap vectors from CSV
    decap_vectors = read_decap_csv(csv_path, max_samples=max_samples, verbose=verbose)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\nOutput directory: {output_dir}")
        print(f"Creating {len(decap_vectors)} occupancy vectors...")
        print(f"Vector length: {decap_vectors.shape[1]}")
        print(f"Starting from sample_{start_index}.npy\n")
    
    # Save each row as an occupancy vector
    success_count = 0
    failed_samples = []
    
    iterator = tqdm(enumerate(decap_vectors), total=len(decap_vectors), 
                   desc="Converting", disable=not verbose)
    
    for idx, decap_vector in iterator:
        try:
            # Save decap vector directly (1D)
            sample_idx = start_index + idx
            output_path = output_dir / f"sample_{sample_idx}.npy"
            np.save(output_path, decap_vector)
            
            success_count += 1
            
        except Exception as e:
            failed_samples.append((idx, str(e)))
            if verbose:
                print(f"\nError processing sample {idx}: {e}")
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Conversion complete!")
        print(f"  Successfully created: {success_count}/{len(decap_vectors)} occupancy vectors")
        if failed_samples:
            print(f"  Failed: {len(failed_samples)} samples")
            for idx, error in failed_samples[:5]:  # Show first 5 failures
                print(f"    Sample {idx}: {error}")
        print(f"  Output directory: {output_dir}")
        print(f"{'='*60}")
    
    return success_count


def get_existing_sample_count(output_dir):
    """Count existing sample_*.npy files in directory."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0
    return len(sorted(output_dir.glob("sample_*.npy")))
