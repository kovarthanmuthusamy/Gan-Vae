"""
Calculate normalization statistics (min/max, percentiles) for the three-branch dataset.

This script computes statistics from raw pre-processed data and saves them to 
normalization_stats.json for use in the normalization pipeline.

Statistics computed:
- Global min/max for each branch
- Percentile min/max (5th and 95th by default)
- File counts

Output: datasets/data_norm/normalization_stats.json
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def calculate_stats(data_dir, percentile_lower=1, percentile_upper=99):
    """
    Calculate normalization statistics from raw data.
    
    Args:
        data_dir: Directory containing heatmap, Imp, Occ_map subdirectories
        percentile_lower: Lower percentile (default 1)
        percentile_upper: Upper percentile (default 99)
    
    Returns:
        Dictionary with statistics for each modality
    """
    
    stats = {
        "percentile_lower": percentile_lower,
        "percentile_upper": percentile_upper,
        "global_min_max": {},
        "percentile_min_max": {}
    }
    
    # ==================== HEATMAP STATISTICS ====================
    heatmap_dir = data_dir / "heatmap"
    if heatmap_dir.exists():
        print("\n[HEATMAP] Calculating statistics...")
        heatmap_files = sorted(heatmap_dir.glob("*.npy"))
        heatmap_count = len(heatmap_files)
        
        # Extract channel 0 (impedance channel) from all files
        heatmap_ch0_values = []
        
        for file_path in tqdm(heatmap_files, desc="Loading heatmap data"):
            try:
                data = np.load(file_path).astype(np.float32)
                # Extract channel 0 (impedance channel)
                if data.shape[0] >= 1:
                    heatmap_ch0_values.append(data[0].flatten())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if heatmap_ch0_values:
            heatmap_ch0_values = np.concatenate(heatmap_ch0_values)
            
            h_global_min = float(np.min(heatmap_ch0_values))
            h_global_max = float(np.max(heatmap_ch0_values))
            h_percentile_min = float(np.percentile(heatmap_ch0_values, percentile_lower))
            h_percentile_max = float(np.percentile(heatmap_ch0_values, percentile_upper))
            
            stats["global_min_max"]["heatmap_count"] = heatmap_count
            stats["global_min_max"]["heatmap_min"] = h_global_min
            stats["global_min_max"]["heatmap_max"] = h_global_max
            
            stats["percentile_min_max"]["heatmap_min"] = h_percentile_min
            stats["percentile_min_max"]["heatmap_max"] = h_percentile_max
            
            print(f"  Heatmap files: {heatmap_count}")
            print(f"  Global range: [{h_global_min:.6f}, {h_global_max:.6f}]")
            print(f"  {percentile_lower}-{percentile_upper}% range: [{h_percentile_min:.6f}, {h_percentile_max:.6f}]")
    
    # ==================== IMPEDANCE STATISTICS (LOG-SCALE) ====================
    imp_dir = data_dir / "Imp"
    if imp_dir.exists():
        print("\n[IMPEDANCE] Calculating statistics (LOG-SCALE)...")
        imp_files = sorted(imp_dir.glob("*.npy"))
        imp_count = len(imp_files)
        
        imp_values = []
        
        for file_path in tqdm(imp_files, desc="Loading impedance data"):
            try:
                data = np.load(file_path).astype(np.float32)
                imp_values.append(data.flatten())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if imp_values:
            imp_values = np.concatenate(imp_values)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            imp_values_safe = np.maximum(imp_values, epsilon)
            
            # Calculate stats in log-space
            log_imp_values = np.log(imp_values_safe)
            
            i_global_min = float(np.min(log_imp_values))
            i_global_max = float(np.max(log_imp_values))
            i_percentile_min = float(np.percentile(log_imp_values, percentile_lower))
            i_percentile_max = float(np.percentile(log_imp_values, percentile_upper))
            
            stats["global_min_max"]["imp_count"] = imp_count
            stats["global_min_max"]["imp_log_min"] = i_global_min
            stats["global_min_max"]["imp_log_max"] = i_global_max
            
            stats["percentile_min_max"]["imp_log_min"] = i_percentile_min
            stats["percentile_min_max"]["imp_log_max"] = i_percentile_max
            
            print(f"  Impedance files: {imp_count}")
            print(f"  Original value range: [{np.min(imp_values):.6f}, {np.max(imp_values):.6f}]")
            print(f"  Log-space global range: [{i_global_min:.6f}, {i_global_max:.6f}]")
            print(f"  Log-space {percentile_lower}-{percentile_upper}% range: [{i_percentile_min:.6f}, {i_percentile_max:.6f}]")
    
    # ==================== OCCUPANCY STATISTICS ====================
    occ_dir = data_dir / "Occ_map"
    if occ_dir.exists():
        print("\n[OCCUPANCY] Calculating statistics...")
        occ_files = sorted(occ_dir.glob("*.npy"))
        occ_count = len(occ_files)
        
        occ_values = []
        
        for file_path in tqdm(occ_files, desc="Loading occupancy data"):
            try:
                data = np.load(file_path).astype(np.float32)
                occ_values.append(data.flatten())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if occ_values:
            occ_values = np.concatenate(occ_values)
            
            o_global_min = float(np.min(occ_values))
            o_global_max = float(np.max(occ_values))
            o_percentile_min = float(np.percentile(occ_values, percentile_lower))
            o_percentile_max = float(np.percentile(occ_values, percentile_upper))
            
            stats["global_min_max"]["occ_count"] = occ_count
            stats["global_min_max"]["occ_min"] = o_global_min
            stats["global_min_max"]["occ_max"] = o_global_max
            
            stats["percentile_min_max"]["occ_min"] = o_percentile_min
            stats["percentile_min_max"]["occ_max"] = o_percentile_max
            
            print(f"  Occupancy files: {occ_count}")
            print(f"  Global range: [{o_global_min:.6f}, {o_global_max:.6f}]")
            print(f"  {percentile_lower}-{percentile_upper}% range: [{o_percentile_min:.6f}, {o_percentile_max:.6f}]")
    
    return stats


def save_stats(stats, output_file):
    """Save statistics to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Statistics saved to: {output_file}")


def display_stats_summary(stats):
    """Display a formatted summary of the calculated statistics."""
    print("\n" + "="*80)
    print("NORMALIZATION STATISTICS SUMMARY")
    print("="*80)
    
    print("\n[GLOBAL MIN/MAX]")
    for key, value in stats["global_min_max"].items():
        print(f"  {key}: {value}")
    
    print(f"\n[PERCENTILE MIN/MAX] ({stats['percentile_lower']}-{stats['percentile_upper']}%)")
    for key, value in stats["percentile_min_max"].items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    # Define paths
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "source" / "data_2"
    output_file = repo_root / "source" / "data_norm" / "normalization_stats.json"
    
    print("\n" + "="*80)
    print("CALCULATING NORMALIZATION STATISTICS")
    print("="*80)
    print(f"\nInput directory: {data_dir}")
    print(f"Output file: {output_file}")
    
    # Check if input directory exists
    if not data_dir.exists():
        print(f"\n✗ Error: Input directory does not exist: {data_dir}")
        return
    
    # Calculate statistics
    stats = calculate_stats(data_dir)
    
    # Display summary
    display_stats_summary(stats)
    
    # Save statistics
    save_stats(stats, output_file)
    
    print("\n✓ Statistics calculation complete!")
    print("\nNext steps:")
    print("  1. Run: python scripts/Normalization.py")
    print("  2. Verify: python scripts/verify_normalization.py")


if __name__ == "__main__":
    main()
