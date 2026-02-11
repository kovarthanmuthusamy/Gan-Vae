import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import argparse
import random
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
TRAIN_DATA_ROOT = SCRIPT_DIR.parent / "source" / "data_2"
OUTPUT_HEATMAP_DIR = TRAIN_DATA_ROOT / "heatmap"
OUTPUT_IMPEDANCE_DIR = TRAIN_DATA_ROOT / "Imp"
OUTPUT_OCCUPANCY_DIR = TRAIN_DATA_ROOT / "Occ_map"

# ============================================================
# INPUT/SOURCE CONFIGURATION
# Specify dataset source and file patterns
# ============================================================
DEFAULT_DATA_ROOT = next((Path(p) for p in [os.getenv("DATA_ROOT"), "/mnt/c/Users/muthusamy/Downloads/DataSet/DATA_1_2_3_4_5", "/mnt/c/Users/muthusamy/Downloads/DataSet"] if p and Path(p).exists()), None)
COMBINATIONS = ["1_true", "2_true", "3_true", "4_true", "5_true"]
MAP_FILE = "Z_0063.000MHz.map"
IMPEDANCE_FILE_SUFFIX = "PIPinZ_IC1_Port1.csv"
DECAPS_FILE = "Decaps.csv"
SKIP_FIRST_DECAP_ROW = False

# ============================================================
# RANDOM SAMPLE SIZE CONFIGURATION FOR EACH COMBINATION
# Set to None to include ALL samples from that combination
# ============================================================
SAMPLES_3_TRUE = 4618  # Number of random samples from 3_true
SAMPLES_4_TRUE = 4618  # Number of random samples from 4_true
SAMPLES_5_TRUE = 4619  # Number of random samples from 5_true
# ============================================================

def _read_csv(filepath, cols=None, **kwargs):
    """Generic CSV reader with caching."""
    try:
        df = pd.read_csv(filepath, **kwargs)
        return df[cols] if cols else df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def _extract_sample_id(path, from_filename=False):
    """Extracts sample ID from path or filename - optimized."""
    if from_filename:
        # Fast check for filename pattern
        if m := re.match(r"(\d+)-", path.name):
            return int(m.group(1))
    # Fast path search without regex for path-based IDs
    for part in path.parts:
        if "PI-" in part:
            try:
                idx = part.index("PI-")
                num_str = part[idx+3:]
                # Extract only leading digits
                digits = ""
                for c in num_str:
                    if c.isdigit():
                        digits += c
                    else:
                        break
                if digits:
                    return int(digits)
            except (ValueError, IndexError):
                pass
    return None

def pair_sample_paths(heatmap_files, impedance_files):
    """Pairs heatmap and impedance files by sample ID."""
    h_dict = {sid: p for p in heatmap_files if (sid := _extract_sample_id(p))}
    i_dict = {sid: p for p in impedance_files if (sid := _extract_sample_id(p, True))}
    matched = sorted(set(h_dict) & set(i_dict))
    return [{"sample_id": sid, "heatmap_path": h_dict[sid], "impedance_path": i_dict[sid]} for sid in matched] if matched else [{"sample_id": None, "heatmap_path": heatmap_files[i], "impedance_path": impedance_files[i]} for i in range(min(len(heatmap_files), len(impedance_files)))]

def iter_dataset_samples(data_root=None, combinations=COMBINATIONS, verbose=True):
    """Iterates through all dataset samples - heavily optimized for speed."""
    from concurrent.futures import ThreadPoolExecutor
    
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    if not root:
        if verbose:
            print("Error: No dataset root found.")
        return
    
    def process_combination(comb):
        """Process a single combination in a thread."""
        try:
            data = _read_csv(root / comb / DECAPS_FILE, header=0)
            if data is None:
                return comb, []
            
            combos = data.select_dtypes(include=[np.number]).values.astype(np.float32)
            combos = combos[1:] if SKIP_FIRST_DECAP_ROW else combos
            comb_root = root / comb
            
            # Use direct path patterns (much faster than recursive glob on huge datasets)
            hm_dir = comb_root / "heatmaps"
            imp_dir = comb_root / "imp"
            
            # Heatmap files are in PI-*/something/{MAP_FILE} - use single level recursion
            heatmap_files = list(hm_dir.glob(f"PI-*/**/{MAP_FILE}")) if hm_dir.exists() else []
            
            # Direct path: PI-*/Power_GND/*PIPinZ... - fast and targeted
            impedance_files = list(imp_dir.glob(f"PI-*/Power_GND/*{IMPEDANCE_FILE_SUFFIX}")) if imp_dir.exists() else []
            
            # Build dictionaries without sorting
            h_dict = {_extract_sample_id(p): p for p in heatmap_files}
            i_dict = {_extract_sample_id(p, True): p for p in impedance_files}
            
            # Remove None keys
            h_dict.pop(None, None)
            i_dict.pop(None, None)
            
            # Find matched sample IDs
            matched_ids = set(h_dict.keys()) & set(i_dict.keys())
            matched_ids = sorted([sid for sid in matched_ids if sid is not None])
            
            samples = []
            for idx, sid in enumerate(matched_ids):
                decap_idx = (sid - 1) if (0 <= sid - 1 < len(combos)) else (idx if idx < len(combos) else None)
                decap_vector = combos[decap_idx] if decap_idx is not None else None
                
                samples.append({
                    "combination": comb,
                    "sample_id": sid,
                    "heatmap_path": h_dict[sid],
                    "impedance_path": i_dict[sid],
                    "decap_vector": decap_vector,
                    "decap_index": decap_idx
                })
            
            if verbose:
                print(f"    Found {len(samples)} samples in {comb}")
            
            return comb, samples
        except Exception as e:
            print(f"  ERROR processing {comb}: {e}")
            import traceback
            traceback.print_exc()
            return comb, []
    
    # Parallel collection using threads (I/O bound)
    print(f"  Scanning combinations in parallel...")
    with ThreadPoolExecutor(max_workers=len(combinations)) as executor:
        futures = [executor.submit(process_combination, comb) for comb in combinations]
        for future in futures:
            comb, samples = future.result()
            for sample in samples:
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

def create_training_dataset(output_root=TRAIN_DATA_ROOT, data_root=None, combinations=COMBINATIONS, 
                            samples_3_true: int | None = SAMPLES_3_TRUE, samples_4_true: int | None = SAMPLES_4_TRUE, 
                            samples_5_true: int | None = SAMPLES_5_TRUE, num_workers=None, verbose=False):
    """Creates training dataset with heatmaps, impedance, and occupancy grids."""
    hm_dir, imp_dir, occ_dir = Path(output_root) / "heatmap", Path(output_root) / "Imp", Path(output_root) / "Occ_map"
    for d in (hm_dir, imp_dir, occ_dir):
        d.mkdir(parents=True, exist_ok=True)
    
    print("Collecting samples...")
    all_samples = list(iter_dataset_samples(data_root, combinations, verbose=True))
    print(f"Found {len(all_samples)} total samples")
    
    # Separate samples by combination
    samples_by_comb = {}
    for sample in all_samples:
        comb = sample["combination"]
        if comb not in samples_by_comb:
            samples_by_comb[comb] = []
        samples_by_comb[comb].append(sample)
    
    print("\nSamples per combination:")
    for comb in combinations:
        count = len(samples_by_comb.get(comb, []))
        print(f"  {comb}: {count} samples")
    
    # Strategy: Include all from 1_true and 2_true, randomly select from others
    selected_samples = []
    
    # Add all from 1_true and 2_true
    for comb in ["1_true", "2_true"]:
        selected_samples.extend(samples_by_comb.get(comb, []))
    
    samples_included = len(selected_samples)
    print(f"\nIncluded all samples from 1_true and 2_true: {samples_included} samples")
    
    # Randomly select from 3_true, 4_true, 5_true based on individual size settings
    random.seed(42)  # For reproducibility
    
    sample_sizes = {
        "3_true": samples_3_true,
        "4_true": samples_4_true,
        "5_true": samples_5_true
    }
    
    for comb, size in sample_sizes.items():
        available = samples_by_comb.get(comb, [])
        if size is None or size >= len(available):
            # Include all samples
            selected_samples.extend(available)
            print(f"  {comb}: Including all {len(available)} samples")
        else:
            # Randomly select specified number
            selected = random.sample(available, size)
            selected_samples.extend(selected)
            print(f"  {comb}: Randomly selected {size} samples (from {len(available)} available)")
    
    print(f"Total selected: {len(selected_samples)} samples")
    
    if not selected_samples:
        return 0
    
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
    parser = argparse.ArgumentParser(description="Create training dataset with heatmaps, impedance, and occupancy grids.")
    parser.add_argument("-o", "--output", type=str, default=str(TRAIN_DATA_ROOT), 
                       help=f"Output directory for dataset (default: {TRAIN_DATA_ROOT})")
    parser.add_argument("-d", "--data-root", type=str, default=None,
                       help="Root directory containing dataset combinations (default: auto-detected)")
    parser.add_argument("-c", "--combinations", type=str, nargs="+", default=COMBINATIONS,
                       help=f"Dataset combinations to process (default: {COMBINATIONS})")
    parser.add_argument("--samples-3", type=int, default=SAMPLES_3_TRUE,
                       help=f"Random samples from 3_true (default: {SAMPLES_3_TRUE}, use -1 for all)")
    parser.add_argument("--samples-4", type=int, default=SAMPLES_4_TRUE,
                       help=f"Random samples from 4_true (default: {SAMPLES_4_TRUE}, use -1 for all)")
    parser.add_argument("--samples-5", type=int, default=SAMPLES_5_TRUE,
                       help=f"Random samples from 5_true (default: {SAMPLES_5_TRUE}, use -1 for all)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                       help="Number of worker processes (default: 4, use 1 for sequential)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATASET CREATION CONFIGURATION")
    print("=" * 60)
    print(f"Output Directory:  {args.output}")
    print(f"Data Root:         {args.data_root or DEFAULT_DATA_ROOT or 'Not found'}")
    print(f"Combinations:      {args.combinations}")
    print(f"Samples 3_true:    {args.samples_3 if args.samples_3 >= 0 else 'All'}")
    print(f"Samples 4_true:    {args.samples_4 if args.samples_4 >= 0 else 'All'}")
    print(f"Samples 5_true:    {args.samples_5 if args.samples_5 >= 0 else 'All'}")
    print(f"Workers:           {args.workers}")
    print(f"Verbose:           {args.verbose}")
    print("=" * 60)
    print()
    
    # Convert -1 to None (meaning all samples)
    s3 = None if args.samples_3 < 0 else args.samples_3
    s4 = None if args.samples_4 < 0 else args.samples_4
    s5 = None if args.samples_5 < 0 else args.samples_5
    
    print("Starting dataset creation...")
    count = create_training_dataset(
        output_root=args.output,
        data_root=args.data_root,
        combinations=args.combinations,
        samples_3_true=s3,
        samples_4_true=s4,
        samples_5_true=s5,
        num_workers=args.workers,
        verbose=args.verbose
    )
    print(f"✓ Processed {count} samples")
    print(f"✓ Output saved to: {args.output}")
