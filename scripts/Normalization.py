import json
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

# Normalization script for heatmap, impedance, and occupancy datasets
# HEATMAP: log(1+x) z-score on foreground (mask=1) pixels, background set to fixed sentinel → 1-channel output
# IMPEDANCE: Log-scale z-score (no clipping)
# OCCUPANCY: Binary data, copied as-is

BACKGROUND_VALUE = None  # Computed dynamically as z_min - 1.5 after stats pass


def calculate_heatmap_stats(heatmap_dir, percentile_lower=0.1, percentile_upper=99.9):
    """Pass 1: Compute log(1+x) mean/std over foreground pixels only."""
    heatmap_dir = Path(heatmap_dir)
    heatmap_files = sorted(heatmap_dir.glob("*.npy"))
    print(f"\n[1/4] Computing heatmap log(1+x) stats from {len(heatmap_files)} files...")

    all_log_fg = []  # collect log(1+x) values for foreground pixels
    for path in tqdm(heatmap_files, desc="Heatmap stats"):
        data = np.load(path).astype(np.float32)
        if data.ndim < 3 or data.shape[0] < 2:
            raise ValueError(f"Unexpected heatmap shape in {path}: {data.shape}")
        ch0, mask = data[0], data[1]
        fg_vals = ch0[mask == 1]  # foreground only
        if fg_vals.size > 0:
            all_log_fg.append(np.log1p(fg_vals))  # log(1 + x)

    all_log_fg = np.concatenate(all_log_fg)
    log_mean = float(all_log_fg.mean())
    log_std = float(all_log_fg.std())
    if log_std == 0.0:
        log_std = 1.0

    print(f"  Foreground log(1+x) stats: mean={log_mean:.4f}, std={log_std:.4f}")
    print(f"  Foreground log(1+x) range: [{all_log_fg.min():.4f}, {all_log_fg.max():.4f}]")
    z_scores = (all_log_fg - log_mean) / log_std
    z_min = float(z_scores.min())
    z_max = float(z_scores.max())
    bg_value = round(z_min - 1.5, 4)
    print(f"  Foreground z-score range: [{z_min:.4f}, {z_max:.4f}]")
    print(f"  Background sentinel value: {bg_value} (z_min - 1.5)")

    return {"log_mean": log_mean, "log_std": log_std,
            "fg_pixel_count": int(all_log_fg.size),
            "z_min": z_min, "z_max": z_max,
            "background_value": bg_value}


def normalize_heatmaps(heatmap_dir, output_heatmap_dir, heatmap_stats):
    """Pass 2: Apply log(1+x) z-score to foreground, sentinel to background, save 1-channel."""
    heatmap_dir, output_heatmap_dir = Path(heatmap_dir), Path(output_heatmap_dir)
    output_heatmap_dir.mkdir(parents=True, exist_ok=True)
    heatmap_files = sorted(heatmap_dir.glob("*.npy"))
    print(f"\n[2/4] Normalizing {len(heatmap_files)} heatmaps (1-channel log(1+x) z-score)...")

    log_mean = heatmap_stats["log_mean"]
    log_std = heatmap_stats["log_std"]
    bg_value = heatmap_stats["background_value"]

    for path in tqdm(heatmap_files, desc="Heatmaps"):
        data = np.load(path).astype(np.float32)
        ch0, mask = data[0], data[1]

        # Apply log(1+x) z-score to foreground
        log_ch0 = np.log1p(ch0)
        z_ch0 = (log_ch0 - log_mean) / log_std

        # Set background to sentinel
        z_ch0[mask == 0] = bg_value

        # Save as 1-channel: (1, 64, 64)
        np.save(output_heatmap_dir / path.name, z_ch0[np.newaxis].astype(np.float32))


def calculate_impedance_stats(imp_dir):
    """Calculate impedance log z-score statistics (no clipping — all values preserved)."""
    imp_dir = Path(imp_dir)
    stats = {}
    if not imp_dir.exists():
        return stats

    imp_files = sorted(imp_dir.glob("*.npy"))
    imp_values = np.concatenate([np.load(f).flatten() for f in tqdm(imp_files, desc="Impedance stats")])

    log_imp = np.log(np.maximum(imp_values, 1e-10))
    i_mean = float(log_imp.mean())
    i_std = float(log_imp.std())
    if i_std == 0.0:
        i_std = 1.0

    z_scores = (log_imp - i_mean) / i_std

    stats.update({
        "imp_count": len(imp_files),
        "log_mean": i_mean,
        "log_std": i_std,
        "z_min": float(z_scores.min()),
        "z_max": float(z_scores.max()),
    })
    print(f"  Impedance log z-score: mean={i_mean:.4f}, std={i_std:.4f}")
    print(f"  z-score range (no clipping): [{z_scores.min():.4f}, {z_scores.max():.4f}]")
    return stats


def normalize_impedance(imp_dir, output_imp_dir, imp_stats):
    """Normalize impedance using log-scale z-score and add derivative as second channel.

    Output shape: (2, 231)
      Ch0: z-score of log(impedance)          — raw signal
      Ch1: first difference of Ch0            — d[i] = z[i+1] - z[i]
           (last element padded with d[-2] to keep constant length)
    """
    imp_dir, output_imp_dir = Path(imp_dir), Path(output_imp_dir)
    output_imp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/4] Normalizing impedance (dual-channel: raw z-score + first derivative)...")

    imp_log_mean = imp_stats["log_mean"]
    imp_log_std = imp_stats["log_std"]

    for path in tqdm(sorted(imp_dir.glob("*.npy")), desc="Impedance"):
        log_data = np.log(np.maximum(np.load(path).astype(np.float32), 1e-10))
        z = ((log_data - imp_log_mean) / imp_log_std).flatten()  # Ch0: (231,)

        # Ch1: first difference  d[i] = z[i+1] - z[i], length kept at 231
        # np.diff gives 230 values; pad the last element by repeating d[-1]
        diff = np.diff(z)                              # (230,)
        deriv = np.append(diff, diff[-1])              # (231,)  last value repeated

        dual = np.stack([z, deriv], axis=0)            # (2, 231)
        np.save(output_imp_dir / path.name, dual)


def copy_occupancy(occ_dir, output_occ_dir):
    """Copy occupancy maps (binary, no normalization needed)."""
    occ_dir, output_occ_dir = Path(occ_dir), Path(output_occ_dir)
    output_occ_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[4/4] Copying occupancy files...")
    for path in tqdm(sorted(occ_dir.glob("*.npy")), desc="Occupancy"):
        np.save(output_occ_dir / path.name, np.load(path).astype(np.float32))


def main():
    """Main function: 4-step normalization pipeline."""
    root = Path(__file__).resolve().parents[1]
    data_root = root / "datasets" / "data_up5"
    output_root = root / "datasets" / "data_up5_norm"

    heatmap_dir = data_root / "heatmap"
    imp_dir = data_root / "Imp"
    occ_dir = data_root / "Occ_map"
    out_heatmap = output_root / "heatmap"
    out_imp = output_root / "Imp"
    out_occ = output_root / "Occ_map"
    stats_file = output_root / "normalization_stats.json"

    print(f"\n{'='*60}\nNORMALIZATION PIPELINE\n{'='*60}")
    print(f"Input: {data_root}\nOutput: {output_root}")

    if not data_root.exists():
        return print(f"\nError: Input directory does not exist: {data_root}")

    # Step 1: Compute heatmap stats (foreground-only log(1+x))
    heatmap_stats = calculate_heatmap_stats(heatmap_dir)

    # Step 2: Normalize heatmaps → 1-channel, log(1+x) z-score, bg=sentinel
    normalize_heatmaps(heatmap_dir, out_heatmap, heatmap_stats)

    # Step 3: Compute impedance stats & normalize
    imp_stats = calculate_impedance_stats(imp_dir)
    normalize_impedance(imp_dir, out_imp, imp_stats)

    # Step 4: Copy occupancy
    copy_occupancy(occ_dir, out_occ)

    # Build and save stats JSON
    stats = {
        "background_value": heatmap_stats["background_value"],
        "Heatmap": heatmap_stats,
        "Impedance": imp_stats,
    }
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}\n✓ NORMALIZATION COMPLETE\n{'='*60}")
    print(f"Dataset: {output_root}\nStats: {stats_file}")


if __name__ == "__main__":
    main()