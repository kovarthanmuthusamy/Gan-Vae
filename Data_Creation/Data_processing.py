import pandas as pd
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

from heatmap import create_Heatmaps, load_mask_board, HEATMAP_GRID_SIZE
from occupancy import create_occupancy_grid, OCC_GRID_H, OCC_GRID_W
from impedance import read_impedance_file, EXPECTED_IMP_LENGTH

# ============================================================
# BASE PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
FRAME_PATH = SCRIPT_DIR.parent / "configs" / "binary_mask.npy"

# ============================================================
# OUTPUT CONFIGURATION
# Specify where to save the processed dataset
# ============================================================
TRAIN_DATA_ROOT = SCRIPT_DIR.parent / "datasets" / "data"
OUTPUT_HEATMAP_DIR = TRAIN_DATA_ROOT / "heatmap"
OUTPUT_IMPEDANCE_DIR = TRAIN_DATA_ROOT / "Imp"
OUTPUT_OCCUPANCY_DIR = TRAIN_DATA_ROOT / "Occ_map"

# ============================================================
# INPUT/SOURCE CONFIGURATION
# Specify dataset source and file patterns
# ============================================================
# New raw data structure: Raw folder contains Heatmap/, Imp/, and decap_combination/ folders
DEFAULT_DATA_ROOT = next((Path(p) for p in [
    os.getenv("DATA_ROOT"),
    "/mnt/c/Users/muthusamy/Desktop/Raw",
    "C:/Users/muthusamy/Desktop/Raw",
    "/home/ubuntu/gan/raw_data"
] if p and Path(p).exists()), None)

# ============================================================
# RANDOM SAMPLE SIZE CONFIGURATION
# Set to None to include ALL samples
# ============================================================
MAX_SAMPLES = 15000  # Set to a number to limit samples, or None for all
# ============================================================

def _read_csv(filepath, cols=None, **kwargs):
    """Generic CSV reader with caching."""
    try:
        df = pd.read_csv(filepath, **kwargs)
        return df[cols] if cols else df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def iter_dataset_samples(data_root=None, verbose=True):
    """
    Iterates through all dataset samples from the new raw data structure.
    
    New structure:
    - Raw/
        - Heatmap/  (contains heatmap files)
        - Imp/      (contains impedance files)
        - decap_combination/  (contains CSV with decap vectors as rows)
    
    Files are matched by order: 1st heatmap → 1st impedance → 1st CSV row
    """
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    if not root or not root.exists():
        if verbose:
            print(f"Error: Dataset root not found: {root}")
        return
    
    # Define subdirectories
    heatmap_dir = root / "Heatmap"
    imp_dir = root / "Imp"
    decap_dir = root / "decap_combinations"
    
    # Validate directories exist
    if not heatmap_dir.exists():
        print(f"Error: Heatmap directory not found: {heatmap_dir}")
        return
    if not imp_dir.exists():
        print(f"Error: Impedance directory not found: {imp_dir}")
        return
    if not decap_dir.exists():
        print(f"Error: Decap combination directory not found: {decap_dir}")
        return
    
    # Helper function to extract PI folder number
    def extract_pi_number(path):
        """Extract number from PI-* folder in path"""
        for part in path.parts:
            if part.startswith('PI-'):
                try:
                    return int(part.split('-')[1])
                except (ValueError, IndexError):
                    pass
        return None
    
    # Get all heatmap files recursively (PI-*/PowerGround/*.map or similar)
    heatmap_all_files = [f for f in heatmap_dir.rglob('*') if f.is_file() and f.suffix.lower() == '.map']
    
    # Get all impedance files recursively (PI-*/PowerGround/*.csv) - only those with IC1 in name
    impedance_all_files = [f for f in imp_dir.rglob('*.csv') if f.is_file() and 'IC1' in f.name]
    
    # Build dictionaries keyed by PI number
    heatmap_dict = {}
    for f in heatmap_all_files:
        pi_num = extract_pi_number(f)
        if pi_num is not None:
            heatmap_dict[pi_num] = f
    
    impedance_dict = {}
    for f in impedance_all_files:
        pi_num = extract_pi_number(f)
        if pi_num is not None:
            impedance_dict[pi_num] = f
    
    # Get sorted PI numbers that exist in both heatmap and impedance
    common_pi_numbers = sorted(set(heatmap_dict.keys()) & set(impedance_dict.keys()))
    
    # Get decap combination CSV file(s)
    decap_csv_files = sorted([f for f in decap_dir.iterdir() if f.is_file() and f.suffix.lower() == '.csv'])
    
    if not decap_csv_files:
        print(f"Error: No CSV files found in {decap_dir}")
        return
    
    # Read decap combinations from the first CSV file
    decap_csv = decap_csv_files[0]
    try:
        decap_data = pd.read_csv(decap_csv, header=None)  # No header, pure data rows
        decap_vectors = decap_data.select_dtypes(include=[np.number]).values.astype(np.float32)
        if verbose:
            print(f"  Loaded {len(decap_vectors)} decap vectors from {decap_csv.name}")
    except Exception as e:
        print(f"Error reading decap CSV {decap_csv}: {e}")
        return
    
    if verbose:
        print(f"  Found {len(heatmap_dict)} heatmap PI folders")
        print(f"  Found {len(impedance_dict)} impedance PI folders")
        print(f"  Found {len(common_pi_numbers)} matching PI folders")
        print(f"  Found {len(decap_vectors)} decap vectors")
    
    # Match by order - use minimum count
    num_samples = min(len(common_pi_numbers), len(decap_vectors))
    
    if verbose:
        print(f"  Matching {num_samples} samples by PI folder order")
    
    # Generate samples matched by PI folder index
    for idx in range(num_samples):
        pi_num = common_pi_numbers[idx]
        sample = {
            "sample_id": idx + 1,
            "pi_number": pi_num,
            "heatmap_path": heatmap_dict[pi_num],
            "impedance_path": impedance_dict[pi_num],
            "decap_vector": decap_vectors[idx],
            "decap_index": idx
        }
        yield sample

_WORKER_MASK_BOARD = None

def _init_worker(frame_path):
    global _WORKER_MASK_BOARD
    try:
        _WORKER_MASK_BOARD = load_mask_board(frame_path)
    except Exception as e:
        print(f"[Worker] ERROR: Failed to load mask board: {e}")
        raise

def _process_sample(task):
    """Processes a single sample with heatmap, impedance, and occupancy grid."""
    idx, sample, hm_dir, imp_dir, occ_dir, verbose = task
    try:
        stacked = create_Heatmaps(sample["heatmap_path"], mask_board=_WORKER_MASK_BOARD, verbose=False)
        imp = read_impedance_file(sample["impedance_path"])
        if imp is None:
            return (idx, False, f"Failed to read impedance")
        if len(imp) != EXPECTED_IMP_LENGTH:
            return (idx, False, f"Invalid impedance length: got {len(imp)}, expected {EXPECTED_IMP_LENGTH}")
        name = f"sample_{idx + 1}.npy"
        np.save(hm_dir / name, stacked)
        np.save(imp_dir / name, imp.reshape(-1, 1))
        if sample["decap_vector"] is not None:
            occ_grid = create_occupancy_grid(sample["decap_vector"])
            np.save(occ_dir / name, occ_grid)
        return (idx, True, None)
    except Exception as e:
        return (idx, False, str(e)[:100])

def create_training_dataset(output_root=TRAIN_DATA_ROOT, data_root=None, 
                            max_samples: int | None = MAX_SAMPLES, 
                            num_workers=None, verbose=False):
    """Creates training dataset with heatmaps, impedance, and occupancy grids."""
    hm_dir, imp_dir, occ_dir = Path(output_root) / "heatmap", Path(output_root) / "Imp", Path(output_root) / "Occ_map"
    for d in (hm_dir, imp_dir, occ_dir):
        d.mkdir(parents=True, exist_ok=True)
    
    print("Collecting samples from new raw data structure...")
    all_samples = list(iter_dataset_samples(data_root, verbose=True))
    print(f"Found {len(all_samples)} total samples")
    
    if not all_samples:
        print("No samples found!")
        return 0
    
    # Optionally limit samples with smart sampling strategy
    if max_samples is not None and max_samples < len(all_samples):
        print(f"Limiting to {max_samples} samples (out of {len(all_samples)} available)")
        
        # Strategy: Always include first 54 and last 54, randomly sample from middle
        FIRST_N = 54
        LAST_N = 54
        
        if max_samples <= FIRST_N + LAST_N:
            # If max_samples is too small, just take first max_samples
            print(f"  Max samples ({max_samples}) <= {FIRST_N + LAST_N}, taking first {max_samples} samples")
            selected_samples = all_samples[:max_samples]
        else:
            # Take first 54
            first_samples = all_samples[:FIRST_N]
            print(f"  Including first {FIRST_N} samples")
            
            # Take last 54
            last_samples = all_samples[-LAST_N:]
            print(f"  Including last {LAST_N} samples")
            
            # Calculate how many to sample from middle
            middle_samples_needed = max_samples - FIRST_N - LAST_N
            
            # Middle samples (excluding first 54 and last 54)
            middle_samples = all_samples[FIRST_N:-LAST_N]
            
            if middle_samples_needed >= len(middle_samples):
                # Include all middle samples
                print(f"  Including all {len(middle_samples)} middle samples")
                selected_middle = middle_samples
            else:
                # Randomly sample from middle
                import random
                random.seed(42)  # For reproducibility
                selected_middle = random.sample(middle_samples, middle_samples_needed)
                print(f"  Randomly sampled {middle_samples_needed} samples from {len(middle_samples)} middle samples")
            
            # Combine: first + middle + last
            selected_samples = first_samples + selected_middle + last_samples
            print(f"  Total selected: {len(selected_samples)} samples (first {FIRST_N} + middle {len(selected_middle)} + last {LAST_N})")
    else:
        selected_samples = all_samples
        print(f"Processing all {len(selected_samples)} samples")
    
    # Create tasks with sequential indices
    tasks = [(i, s, hm_dir, imp_dir, occ_dir, verbose) for i, s in enumerate(selected_samples)]
    
    # Use multiprocessing for faster processing
    num_workers = num_workers if num_workers and num_workers > 0 else min(4, os.cpu_count() or 1)
    print(f"\nRunning with {num_workers} workers...")
    results = []
    
    if num_workers == 1:
        # Sequential processing
        _init_worker(FRAME_PATH)
        for i, task in enumerate(tasks):
            idx, success, error = _process_sample(task)
            results.append(success)
            
            if success:
                print(f"  ✓ Created sample_{idx + 1}")
            else:
                print(f"  ✗ sample_{idx + 1}: {error}")
            
            if (i + 1) % max(1, len(tasks) // 10) == 0 or i + 1 == len(tasks):
                print(f"  Progress: {i + 1}/{len(tasks)} samples processed")
    else:
        # Parallel processing
        try:
            from multiprocessing import Pool
            with Pool(processes=num_workers, initializer=_init_worker, initargs=(FRAME_PATH,)) as pool:
                print(f"Processing {len(tasks)} samples in parallel...")
                completed = 0
                for idx, success, error in pool.imap_unordered(_process_sample, tasks):
                    results.append(success)
                    completed += 1
                    
                    if success:
                        print(f"  ✓ Created sample_{idx + 1}")
                    else:
                        print(f"  ✗ sample_{idx + 1}: {error}")
                    
                    if completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks):
                        print(f"  Progress: {completed}/{len(tasks)} samples processed")
        except Exception as e:
            print(f"ERROR in parallel processing: {e}")
            print("Falling back to sequential processing...")
            _init_worker(FRAME_PATH)
            for i, task in enumerate(tasks):
                idx, success, error = _process_sample(task)
                results.append(success)
                if (i + 1) % max(1, len(tasks) // 10) == 0 or i + 1 == len(tasks):
                    print(f"  Progress: {i + 1}/{len(tasks)} samples processed")
    
    success_count = sum(1 for r in results if r)
    return success_count

if __name__ == "__main__":
    print("=" * 60)
    print("DATASET CREATION CONFIGURATION")
    print("=" * 60)
    print(f"Output Directory:  {TRAIN_DATA_ROOT}")
    print(f"Data Root:         {DEFAULT_DATA_ROOT or 'Not found'}")
    print(f"Max Samples:       {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print(f"Workers:           4")
    print(f"Verbose:           False")
    print("=" * 60)
    print()
    
    print("Starting dataset creation...")
    count = create_training_dataset(
        output_root=TRAIN_DATA_ROOT,
        data_root=None,
        max_samples=MAX_SAMPLES,
        num_workers=4,
        verbose=False
    )
    print(f"✓ Processed {count} samples")
    print(f"✓ Output saved to: {TRAIN_DATA_ROOT}")
