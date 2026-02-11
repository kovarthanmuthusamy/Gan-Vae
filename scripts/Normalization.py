import json
import numpy as np
from pathlib import Path

#==============================================================================
# Normalization script for heatmap, impedance, and occupancy datasets
# - Heatmap: Normalize channel 0 (impedance) using linear min-max, keep channel 1 (mask) unchanged
# - Impedance: Normalize entire 231-element vector using LOG-SCALE min-max normalization
# - Occupancy: Binary data, no normalization needed (just copy)
#
# NOTE: Impedance uses log-scale normalization because impedance values vary over
#       multiple orders of magnitude (logarithmic distribution). This ensures better
#       distribution of normalized values across [0, 1] range.
#==============================================================================

# Initialize normalization parameters (these will be loaded from the stats file)
norm_type = "percentile_min_max"  # Options: "percentile_min_max" or "global_min_max"

def normalize_dataset(heatmap_dir, imp_dir, occ_dir, output_root, heatmap_occ_stats, imp_stats):
    heatmap_dir = Path(heatmap_dir)
    imp_dir = Path(imp_dir)
    occ_dir = Path(occ_dir)
    output_root = Path(output_root)

    out_heatmap = output_root / "heatmap"
    out_imp = output_root / "Imp"
    out_occ = output_root / "Occ_map"
    out_heatmap.mkdir(parents=True, exist_ok=True)
    out_imp.mkdir(parents=True, exist_ok=True)
    out_occ.mkdir(parents=True, exist_ok=True)

    heatmap_min = heatmap_occ_stats["heatmap_min"]
    heatmap_max = heatmap_occ_stats["heatmap_max"]
    imp_log_min = imp_stats["imp_log_min"]
    imp_log_max = imp_stats["imp_log_max"]

    # Normalize heatmap (impedance channel)
    heatmap_files = sorted(heatmap_dir.glob("*.npy"))
    for path in heatmap_files:
        data = np.load(path).astype(np.float32)
        if data.ndim < 3 or data.shape[0] < 2:
            raise ValueError(f"Unexpected heatmap shape in {path}: {data.shape}")
        # Normalize channel 0 (impedance)
        channel0 = data[0]
        channel0 = np.clip(channel0, heatmap_min, heatmap_max)
        denom = heatmap_max - heatmap_min
        if denom == 0:
            data[0] = 0.0
        else:
            data[0] = (channel0 - heatmap_min) / denom
        # Channel 1 (mask) stays as is
        np.save(out_heatmap / path.name, data)

    # Normalize impedance (231-element vector) using LOG-SCALE
    imp_files = sorted(imp_dir.glob("*.npy"))
    epsilon = 1e-10
    for path in imp_files:
        data = np.load(path).astype(np.float32)
        # Apply log transform
        data_safe = np.maximum(data, epsilon)
        log_data = np.log(data_safe)
        # Clip and normalize in log-space
        log_data = np.clip(log_data, imp_log_min, imp_log_max)
        denom = imp_log_max - imp_log_min
        if denom == 0:
            norm = np.zeros_like(log_data, dtype=np.float32)
        else:
            norm = (log_data - imp_log_min) / denom
        np.save(out_imp / path.name, norm)

    # Copy occupancy (binary - no normalization needed)
    occ_files = sorted(occ_dir.glob("*.npy"))
    for path in occ_files:
        data = np.load(path).astype(np.float32)
        np.save(out_occ / path.name, data)

def main():
    data_root = Path(__file__).resolve().parents[1] / "datasets" / "source" / "data_2"
    heatmap_dir = data_root /  "heatmap"
    imp_dir = data_root / "Imp"
    occ_dir = data_root / "Occ_map"
    output_root = Path(__file__).resolve().parents[1] / "datasets" / "source" / "data_norm"
    stats_file = Path(__file__).resolve().parents[1] / "datasets" / "source" / "data_norm" / "normalization_stats.json"
    
    with open(stats_file, "r") as f:
        stats_data = json.load(f)
    
    normalization_stats = stats_data[norm_type]
    
    # Use 1-99 percentile for heatmap/occupancy (balance outlier removal & space utilization)
    # Use 1-99 percentile in log-space for impedance (better distribution for logarithmic data)
    heatmap_occ_stats = normalization_stats
    imp_stats = normalization_stats
    print(f"Heatmap files: {normalization_stats.get('heatmap_count', 'N/A')}")
    print(f"Impedance files: {normalization_stats.get('imp_count', 'N/A')}")
    print(f"Occupancy files: {normalization_stats.get('occ_count', 'N/A')}")
    print("The min/max values for normalization:")
    print(
        f"Heatmap channel 0 min/max (1-99 percentile): {heatmap_occ_stats['heatmap_min']} / {heatmap_occ_stats['heatmap_max']}"
    )
    print(f"Impedance log-min/log-max (1-99 percentile): {imp_stats['imp_log_min']} / {imp_stats['imp_log_max']}")
    ## call the normalization function
    normalize_dataset(heatmap_dir, imp_dir, occ_dir, output_root, heatmap_occ_stats, imp_stats)
    print(f"Normalized dataset saved to {output_root}")




if __name__ == "__main__":
    main()
