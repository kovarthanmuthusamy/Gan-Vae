import json
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

# Normalization script for heatmap, impedance, and occupancy datasets
# HEATMAP: Per-sample min-max normalization (each divided by its max absolute value) → [0,1]
# MAX_VALUE: Extracted from heatmaps, percentile min-max normalized for VAE training
# IMPEDANCE: Log-scale min-max normalization using percentile ranges
# OCCUPANCY: Binary data, copied as-is

def normalize_heatmaps_and_extract_max_values(heatmap_dir, output_heatmap_dir):
    """Normalize heatmaps using per-sample min-max and extract raw max values."""
    heatmap_dir, output_heatmap_dir = Path(heatmap_dir), Path(output_heatmap_dir)
    output_heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    heatmap_files = sorted(heatmap_dir.glob("*.npy"))
    print(f"\n[1/4] Normalizing {len(heatmap_files)} heatmaps...")
    
    max_values_raw = []
    for path in tqdm(heatmap_files, desc="Heatmaps"):
        data = np.load(path).astype(np.float32)
        if data.ndim < 3 or data.shape[0] < 2:
            raise ValueError(f"Unexpected heatmap shape in {path}: {data.shape}")
        
        max_value = max(np.abs(data[0]).max(), 1e-8)
        max_values_raw.append(max_value)
        data[0] = np.clip(data[0] / max_value, 0.0, 1.0)
        np.save(output_heatmap_dir / path.name, data)
    
    return np.array(max_values_raw)

def calculate_stats(imp_dir, max_values_raw, percentile_lower=1, percentile_upper=99):
    """Calculate normalization statistics for impedance and max_values."""
    imp_dir = Path(imp_dir)
    stats = {"percentile_lower": percentile_lower, "percentile_upper": percentile_upper, 
             "global_min_max": {}, "percentile_min_max": {}}
    
    print(f"\n[2/4] Calculating statistics...")
    
    # IMPEDANCE - LOG-SCALE
    if imp_dir.exists():
        imp_files = sorted(imp_dir.glob("*.npy"))
        imp_values = np.concatenate([np.load(f).flatten() for f in tqdm(imp_files, desc="Impedance")])
        
        log_imp = np.log(np.maximum(imp_values, 1e-10))
        i_global_min, i_global_max = float(log_imp.min()), float(log_imp.max())
        i_perc_min, i_perc_max = float(np.percentile(log_imp, percentile_lower)), float(np.percentile(log_imp, percentile_upper))
        
        for key in ["global_min_max", "percentile_min_max"]:
            stats[key].update({"imp_count": len(imp_files), "imp_log_min": i_perc_min if "percentile" in key else i_global_min,
                               "imp_log_max": i_perc_max if "percentile" in key else i_global_max})
        
        print(f"  Impedance: {len(imp_files)} files, log-range [{i_perc_min:.4f}, {i_perc_max:.4f}]")
    
    # MAX_VALUE - PERCENTILE MIN-MAX
    mv_global_min, mv_global_max = float(max_values_raw.min()), float(max_values_raw.max())
    mv_perc_min, mv_perc_max = float(np.percentile(max_values_raw, percentile_lower)), float(np.percentile(max_values_raw, percentile_upper))
    
    for key in ["global_min_max", "percentile_min_max"]:
        stats[key].update({"max_value_min": mv_perc_min if "percentile" in key else mv_global_min,
                          "max_value_max": mv_perc_max if "percentile" in key else mv_global_max})
    
    print(f"  Max values: {len(max_values_raw)} values, range [{mv_perc_min:.4f}, {mv_perc_max:.4f}]")
    return stats

def normalize_impedance_and_max_values(imp_dir, max_values_raw, heatmap_files, 
                                       output_imp_dir, output_max_value_dir, stats):
    """Normalize impedance using log-scale and max_values using percentile min-max."""
    imp_dir, output_imp_dir, output_max_value_dir = Path(imp_dir), Path(output_imp_dir), Path(output_max_value_dir)
    output_imp_dir.mkdir(parents=True, exist_ok=True)
    output_max_value_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[3/4] Applying normalization...")
    
    # Impedance - LOG-SCALE
    imp_log_min, imp_log_max = stats["percentile_min_max"]["imp_log_min"], stats["percentile_min_max"]["imp_log_max"]
    denom = imp_log_max - imp_log_min if imp_log_max != imp_log_min else 1.0
    
    for path in tqdm(sorted(imp_dir.glob("*.npy")), desc="Impedance"):
        log_data = np.log(np.maximum(np.load(path).astype(np.float32), 1e-10))
        norm = (np.clip(log_data, imp_log_min, imp_log_max) - imp_log_min) / denom
        np.save(output_imp_dir / path.name, norm)
    
    # Max_values - PERCENTILE MIN-MAX
    mv_min, mv_max = stats["percentile_min_max"]["max_value_min"], stats["percentile_min_max"]["max_value_max"]
    mv_denom = mv_max - mv_min if mv_max != mv_min else 1.0
    for i, path in enumerate(tqdm(heatmap_files, desc="Max values")):
        clipped = np.clip(max_values_raw[i], mv_min, mv_max)
        max_val_norm = np.array([(clipped - mv_min) / mv_denom], dtype=np.float32)
        np.save(output_max_value_dir / path.name, max_val_norm)

def copy_occupancy(occ_dir, output_occ_dir):
    """Copy occupancy maps (binary, no normalization needed)."""
    occ_dir, output_occ_dir = Path(occ_dir), Path(output_occ_dir)
    output_occ_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[4/4] Copying occupancy files...")
    for path in tqdm(sorted(occ_dir.glob("*.npy")), desc="Occupancy"):
        np.save(output_occ_dir / path.name, np.load(path).astype(np.float32))

def main():
    """Main function: 4-step normalization pipeline."""
    # Define paths
    root = Path(__file__).resolve().parents[1]
    data_root = root / "datasets" / "data"
    output_root = root / "datasets" / "data_norm"
    
    heatmap_dir, imp_dir, occ_dir = data_root / "heatmap", data_root / "Imp", data_root / "Occ_map"
    out_heatmap, out_imp, out_occ, out_max_value = (output_root / "heatmap", output_root / "Imp", 
                                                      output_root / "Occ_map", output_root / "Max_value")
    stats_file = output_root / "normalization_stats.json"
    
    print(f"\n{'='*60}\nNORMALIZATION PIPELINE\n{'='*60}")
    print(f"Input: {data_root}\nOutput: {output_root}")
    
    if not data_root.exists():
        return print(f"\nError: Input directory does not exist: {data_root}")
    
    # Execute pipeline
    max_values_raw = normalize_heatmaps_and_extract_max_values(heatmap_dir, out_heatmap)
    stats = calculate_stats(imp_dir, max_values_raw)
    normalize_impedance_and_max_values(imp_dir, max_values_raw, sorted(heatmap_dir.glob("*.npy")), 
                                       out_imp, out_max_value, stats)
    copy_occupancy(occ_dir, out_occ)
    
    # Save statistics
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    exp_stats_file = root / "experiments" / "exp013" / "metrics" / "normalization_stats.json"
    exp_stats_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(stats_file, exp_stats_file)
    
    print(f"\n{'='*60}\n✓ NORMALIZATION COMPLETE\n{'='*60}")
    print(f"Dataset: {output_root}\nStats: {stats_file}\n       {exp_stats_file}")


if __name__ == "__main__":
    main()