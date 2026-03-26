import pandas as pd
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

from heatmap import create_Heatmaps, load_mask_board
from impedance import read_impedance_file, EXPECTED_IMP_LENGTH
from csv_to_occupancy import create_occupancy_vector

# ============================================================
# BASE PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
FRAME_PATH = SCRIPT_DIR.parent / "configs" / "binary_mask.npy"

# ============================================================
# OUTPUT CONFIGURATION
# Specify where to save the processed dataset
# ============================================================
TRAIN_DATA_ROOT = SCRIPT_DIR.parent / "datasets" / "data_up5"
OUTPUT_HEATMAP_DIR = TRAIN_DATA_ROOT / "heatmap"
OUTPUT_IMPEDANCE_DIR = TRAIN_DATA_ROOT / "Imp"
OUTPUT_OCCUPANCY_DIR = TRAIN_DATA_ROOT / "Occ_map"

# =========================================================
# INPUT/SOURCE CONFIGURATION
# Specify dataset source and file patterns
# ============================================================
# Dataset structure:
#   Dataset/
#     1_true/  Heatmap/ PI-N/  Imp/ PI-N/  decap  (CSV file)
#     2_true/  ...
#     ...
#     5_true/  ...
DEFAULT_DATA_ROOT = next((Path(p) for p in [
    os.getenv("DATA_ROOT"),
    "/mnt/c/Users/muthusamy/Desktop/DataSet"
] if p and Path(p).exists()), None)

# ============================================================
# RANDOM SAMPLE SIZE CONFIGURATION
# Set to None to include ALL samples
# ============================================================
MAX_SAMPLES = 30000  # Set to a number to limit samples, or None for all
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
    Iterates through all dataset samples from the multi-folder dataset structure.

    Expected structure:
        Dataset/
          1_true/
            Heatmap/   PI-1/  PI-2/  ...   (*.map files inside)
            Imp/       PI-1/  PI-2/  ...   (*IC1*.csv files inside)
            decap                          (CSV file: each row = decap vector for one PI)
          2_true/  ...   5_true/  ...

    Matching logic per sub-folder:
      - PI folders are matched between Heatmap/ and Imp/ by shared PI number.
      - Decap rows are matched to sorted PI numbers by position (row 0 → lowest PI number).
      - Samples are yielded with a globally incremented sample_id.
    """
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    if not root or not root.exists():
        if verbose:
            print(f"Error: Dataset root not found: {root}")
        return

    # Helper: extract integer from a "PI-N" path component
    def extract_pi_number(path):
        for part in Path(path).parts:
            if part.upper().startswith('PI-'):
                try:
                    return int(part.split('-')[1])
                except (ValueError, IndexError):
                    pass
        return None

    # Find all N_true sub-folders, sorted numerically
    true_folders = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.endswith('_true')],
        key=lambda d: int(d.name.split('_')[0])
    )

    if not true_folders:
        print(f"Error: No '*_true' sub-folders found in {root}")
        return

    if verbose:
        print(f"  Found {len(true_folders)} sub-folders: {[d.name for d in true_folders]}")

    global_idx = 0  # globally unique sequential index across all sub-folders

    for folder in true_folders:
        heatmap_dir = folder / "heatmaps"
        imp_dir     = folder / "imp"

        # Locate the decap CSV file directly inside the numbered folder
        decap_candidates = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() == '.csv' and 'decap' in f.name.lower()
        ]
        if not decap_candidates:
            print(f"  Warning: No decap file found in {folder.name}, skipping.")
            continue
        decap_csv = decap_candidates[0]

        # Validate sub-directories
        if not heatmap_dir.exists():
            print(f"  Warning: heatmaps/ not found in {folder.name}, skipping.")
            continue
        if not imp_dir.exists():
            print(f"  Warning: imp/ not found in {folder.name}, skipping.")
            continue

        # Collect heatmap files keyed by PI number
        heatmap_dict = {}
        for f in heatmap_dir.rglob('*'):
            if f.is_file() and f.suffix.lower() == '.map':
                pi_num = extract_pi_number(f)
                if pi_num is not None:
                    heatmap_dict[pi_num] = f

        # Collect impedance files keyed by PI number (IC1 files only)
        impedance_dict = {}
        for f in imp_dir.rglob('*.csv'):
            if f.is_file() and 'IC1' in f.name:
                pi_num = extract_pi_number(f)
                if pi_num is not None:
                    impedance_dict[pi_num] = f

        # PI numbers present in both Heatmap and Imp
        common_pi_numbers = sorted(set(heatmap_dict.keys()) & set(impedance_dict.keys()))

        # Load decap vectors from the CSV (each row = one vector)
        try:
            decap_data = pd.read_csv(decap_csv)  # first row is header (C1,C2,...,C52)
            decap_vectors = decap_data.select_dtypes(include=[np.number]).values.astype(np.float32)
        except Exception as e:
            print(f"  Error reading decap CSV {decap_csv}: {e}, skipping {folder.name}.")
            continue

        num_samples = min(len(common_pi_numbers), len(decap_vectors))

        if verbose:
            print(f"\n  [{folder.name}]")
            print(f"    Heatmap PI folders : {len(heatmap_dict)}")
            print(f"    Imp PI folders     : {len(impedance_dict)}")
            print(f"    Common PI folders  : {len(common_pi_numbers)}")
            print(f"    Decap rows         : {len(decap_vectors)}")
            print(f"    Samples this folder: {num_samples}")

        for i in range(num_samples):
            pi_num = common_pi_numbers[i]
            yield {
                "sample_id":      global_idx + 1,
                "source_folder":  folder.name,
                "pi_number":      pi_num,
                "heatmap_path":   heatmap_dict[pi_num],
                "impedance_path": impedance_dict[pi_num],
                "decap_vector":   decap_vectors[i],
                "decap_index":    i,
            }
            global_idx += 1

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
        stacked = create_Heatmaps(sample["heatmap_path"], mask_board=_WORKER_MASK_BOARD, verbose=True)
        imp = read_impedance_file(sample["impedance_path"])
        if imp is None:
            return (idx, False, f"Failed to read impedance")
        if len(imp) != EXPECTED_IMP_LENGTH:
            return (idx, False, f"Invalid impedance length: got {len(imp)}, expected {EXPECTED_IMP_LENGTH}")
        name = f"sample_{idx + 1}.npy"
        np.save(hm_dir / name, stacked)
        np.save(imp_dir / name, imp.reshape(-1, 1))
        if sample["decap_vector"] is not None:
            occ_vec = create_occupancy_vector(sample["decap_vector"])
            np.save(occ_dir / name, occ_vec)
        return (idx, True, None)
    except Exception as e:
        return (idx, False, str(e)[:100])

def create_training_dataset(output_root=TRAIN_DATA_ROOT, data_root=None, 
                            max_samples: int | None = MAX_SAMPLES, 
                            num_workers=None, verbose=True):
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
        
        # Strategy: Always include first 1150 and last 54, randomly sample from middle
        FIRST_N = 1150
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
    num_workers = num_workers if num_workers and num_workers > 0 else min(48, os.cpu_count() or 1)
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
    import sys
    
    # Check for CSV-only mode as command-line argument
    if len(sys.argv) > 1 and sys.argv[1] == "--csv-only":
        print("\n🔄 CSV-ONLY MODE: Creating occupancy grids from CSV\n")
        
        # Get CSV path from command line or use default
        csv_path = sys.argv[2] if len(sys.argv) > 2 else (
            DEFAULT_DATA_ROOT / "1_true" / "decap"
            if DEFAULT_DATA_ROOT else None
        )
    
    # Normal mode: full dataset creation
    print("=" * 60)
    print("DATASET CREATION CONFIGURATION")
    print("=" * 60)
    print(f"Output Directory:  {TRAIN_DATA_ROOT}")
    print(f"Data Root:         {DEFAULT_DATA_ROOT or 'Not found'}")
    print(f"Max Samples:       {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print(f"Workers:           64")
    print(f"Verbose:           True")
    print("=" * 60)
    print("Starting dataset creation...")
    count = create_training_dataset(
        output_root=TRAIN_DATA_ROOT,
        data_root=None,
        max_samples=MAX_SAMPLES,
        num_workers=64,
        verbose=True
    )
    print(f"✓ Processed {count} samples")
    print(f"✓ Output saved to: {TRAIN_DATA_ROOT}")
