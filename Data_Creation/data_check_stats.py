"""Calculate dataset statistics: mean, median, variance, and check occupancy skewness."""
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm


def calculate_statistics(data_root="datasets/data"):
    """Calculate comprehensive statistics for all modalities in the dataset.
    
    Args:
        data_root: Root directory containing heatmap/, Imp/, Occ_map/ folders
    """
    data_root = Path(data_root)
    
    print("="*80)
    print("DATASET STATISTICS CALCULATOR")
    print("="*80)
    print(f"Data directory: {data_root}\n")
    
    # Find all samples
    heatmap_dir = data_root / "heatmap"
    impedance_dir = data_root / "Imp"
    occupancy_dir = data_root / "Occ_map"
    
    sample_files = sorted(heatmap_dir.glob("sample_*.npy"))
    sample_indices = []
    for f in sample_files:
        try:
            idx = int(f.stem.split('_')[1])
            sample_indices.append(idx)
        except (ValueError, IndexError):
            pass
    
    sample_indices.sort()
    n_samples = len(sample_indices)
    
    print(f"Found {n_samples} samples\n")
    
    if n_samples == 0:
        print("No samples found!")
        return
    
    # Initialize storage for all data
    heatmap_impedance_ch = []
    heatmap_mask_ch = []
    impedance_vectors = []
    occupancy_grids = []
    
    print("Loading all samples...")
    for idx in tqdm(sample_indices):
        # Load heatmap (2 channels: impedance + mask)
        heatmap_file = heatmap_dir / f"sample_{idx}.npy"
        if heatmap_file.exists():
            heatmap = np.load(heatmap_file)
            if heatmap.shape[0] == 2:
                heatmap_impedance_ch.append(heatmap[0])
                heatmap_mask_ch.append(heatmap[1])
        
        # Load impedance vector
        impedance_file = impedance_dir / f"sample_{idx}.npy"
        if impedance_file.exists():
            impedance = np.load(impedance_file).flatten()
            impedance_vectors.append(impedance)
        
        # Load occupancy grid
        occupancy_file = occupancy_dir / f"sample_{idx}.npy"
        if occupancy_file.exists():
            occupancy = np.load(occupancy_file)
            occupancy_grids.append(occupancy)
    
    print("\n" + "="*80)
    print("HEATMAP IMPEDANCE CHANNEL STATISTICS")
    print("="*80)
    if heatmap_impedance_ch:
        heatmap_imp_data = np.array(heatmap_impedance_ch).flatten()
        print(f"Total samples: {len(heatmap_impedance_ch)}")
        print(f"Shape per sample: {heatmap_impedance_ch[0].shape}")
        print(f"Total values: {len(heatmap_imp_data):,}")
        print(f"\nMean:     {heatmap_imp_data.mean():.6f}")
        print(f"Median:   {np.median(heatmap_imp_data):.6f}")
        print(f"Variance: {heatmap_imp_data.var():.6f}")
        print(f"Std Dev:  {heatmap_imp_data.std():.6f}")
        print(f"Min:      {heatmap_imp_data.min():.6f}")
        print(f"Max:      {heatmap_imp_data.max():.6f}")
        print(f"Skewness: {stats.skew(heatmap_imp_data):.6f}")
        print(f"Kurtosis: {stats.kurtosis(heatmap_imp_data):.6f}")
    else:
        print("No heatmap data found!")
    
    print("\n" + "="*80)
    print("HEATMAP MASK CHANNEL STATISTICS")
    print("="*80)
    if heatmap_mask_ch:
        heatmap_mask_data = np.array(heatmap_mask_ch).flatten()
        print(f"Total samples: {len(heatmap_mask_ch)}")
        print(f"Shape per sample: {heatmap_mask_ch[0].shape}")
        print(f"Total values: {len(heatmap_mask_data):,}")
        print(f"\nMean:     {heatmap_mask_data.mean():.6f}")
        print(f"Median:   {np.median(heatmap_mask_data):.6f}")
        print(f"Variance: {heatmap_mask_data.var():.6f}")
        print(f"Std Dev:  {heatmap_mask_data.std():.6f}")
        print(f"Min:      {heatmap_mask_data.min():.6f}")
        print(f"Max:      {heatmap_mask_data.max():.6f}")
        print(f"Unique values: {np.unique(heatmap_mask_data)}")
    else:
        print("No heatmap mask data found!")
    
    print("\n" + "="*80)
    print("IMPEDANCE VECTOR STATISTICS")
    print("="*80)
    if impedance_vectors:
        impedance_data = np.array(impedance_vectors).flatten()
        print(f"Total samples: {len(impedance_vectors)}")
        print(f"Vector length per sample: {impedance_vectors[0].shape[0]}")
        print(f"Total values: {len(impedance_data):,}")
        print(f"\nMean:     {impedance_data.mean():.6f}")
        print(f"Median:   {np.median(impedance_data):.6f}")
        print(f"Variance: {impedance_data.var():.6f}")
        print(f"Std Dev:  {impedance_data.std():.6f}")
        print(f"Min:      {impedance_data.min():.6f}")
        print(f"Max:      {impedance_data.max():.6f}")
        print(f"Skewness: {stats.skew(impedance_data):.6f}")
        print(f"Kurtosis: {stats.kurtosis(impedance_data):.6f}")
    else:
        print("No impedance data found!")
    
    print("\n" + "="*80)
    print("OCCUPANCY GRID STATISTICS & SKEWNESS ANALYSIS")
    print("="*80)
    if occupancy_grids:
        occupancy_data = np.array(occupancy_grids).flatten()
        print(f"Total samples: {len(occupancy_grids)}")
        print(f"Grid shape per sample: {occupancy_grids[0].shape}")
        print(f"Total cells: {len(occupancy_data):,}")
        
        # Basic statistics
        print(f"\nMean:     {occupancy_data.mean():.6f}")
        print(f"Median:   {np.median(occupancy_data):.6f}")
        print(f"Variance: {occupancy_data.var():.6f}")
        print(f"Std Dev:  {occupancy_data.std():.6f}")
        
        # Binary distribution
        unique, counts = np.unique(occupancy_data, return_counts=True)
        print(f"\nValue distribution:")
        for val, count in zip(unique, counts):
            percentage = (count / len(occupancy_data)) * 100
            print(f"  Value {val}: {count:,} ({percentage:.2f}%)")
        
        # Skewness analysis
        skewness = stats.skew(occupancy_data)
        print(f"\n*** SKEWNESS ANALYSIS ***")
        print(f"Skewness: {skewness:.6f}")
        
        if abs(skewness) < 0.5:
            skew_interpretation = "Fairly symmetric"
        elif skewness > 0.5:
            skew_interpretation = "Right-skewed (more 0s, fewer 1s)"
        else:
            skew_interpretation = "Left-skewed (more 1s, fewer 0s)"
        
        print(f"Interpretation: {skew_interpretation}")
        
        # Additional skewness metrics
        n_occupied = np.sum(occupancy_data == 1)
        n_empty = np.sum(occupancy_data == 0)
        occupancy_rate = (n_occupied / len(occupancy_data)) * 100
        
        print(f"\nOccupancy rate: {occupancy_rate:.2f}%")
        print(f"Empty cells: {n_empty:,} ({100-occupancy_rate:.2f}%)")
        print(f"Occupied cells: {n_occupied:,} ({occupancy_rate:.2f}%)")
        
        # Per-sample occupancy distribution
        per_sample_occupancy = []
        for occ_grid in occupancy_grids:
            occupancy_rate_sample = (np.sum(occ_grid == 1) / occ_grid.size) * 100
            per_sample_occupancy.append(occupancy_rate_sample)
        
        per_sample_occupancy = np.array(per_sample_occupancy)
        print(f"\nPer-sample occupancy rate:")
        print(f"  Mean: {per_sample_occupancy.mean():.2f}%")
        print(f"  Std:  {per_sample_occupancy.std():.2f}%")
        print(f"  Min:  {per_sample_occupancy.min():.2f}%")
        print(f"  Max:  {per_sample_occupancy.max():.2f}%")
        
        # Histogram of per-sample occupancy
        print(f"\nDistribution of per-sample occupancy rates:")
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        hist, _ = np.histogram(per_sample_occupancy, bins=bins)
        for i in range(len(hist)):
            print(f"  {bins[i]:3.0f}%-{bins[i+1]:3.0f}%: {hist[i]:4d} samples")
        
        # Check if dataset is imbalanced
        print(f"\n*** IMBALANCE CHECK ***")
        if occupancy_rate < 20 or occupancy_rate > 80:
            print(f"⚠️  WARNING: Dataset is imbalanced (occupancy rate: {occupancy_rate:.2f}%)")
            if occupancy_rate < 20:
                print("   - Very few occupied cells (sparse occupancy)")
                print("   - Consider using weighted loss or focal loss for occupancy")
            else:
                print("   - Very few empty cells (dense occupancy)")
                print("   - Consider using weighted loss for occupancy")
        else:
            print(f"✓ Dataset is reasonably balanced (occupancy rate: {occupancy_rate:.2f}%)")
    else:
        print("No occupancy data found!")
    
    # ============================================================
    # DATASET VARIATION ANALYSIS (Inter-sample diversity)
    # ============================================================
    print("\n" + "="*80)
    print("DATASET VARIATION ANALYSIS (Can the model learn patterns?)")
    print("="*80)
    
    # 1. HEATMAP VARIATION
    if heatmap_impedance_ch:
        print("\n--- HEATMAP VARIATION ---")
        heatmap_samples = np.array(heatmap_impedance_ch)  # Shape: (n_samples, 64, 64)
        
        # Flatten each sample for comparison
        heatmap_flat = heatmap_samples.reshape(len(heatmap_samples), -1)  # (n_samples, 4096)
        
        # Inter-sample variance (variance across samples for each pixel)
        inter_sample_var = np.var(heatmap_flat, axis=0).mean()  # Average variance per pixel across samples
        
        # Intra-sample variance (variance within each sample)
        intra_sample_vars = [np.var(sample) for sample in heatmap_samples]
        avg_intra_var = np.mean(intra_sample_vars)
        
        # Sample-to-sample differences (pairwise distances)
        sample_means = heatmap_flat.mean(axis=1)
        sample_stds = heatmap_flat.std(axis=1)
        
        print(f"  Inter-sample variance: {inter_sample_var:.6f}")
        print(f"    → How much pixels vary across different samples")
        print(f"  Intra-sample variance (avg): {avg_intra_var:.6f}")
        print(f"    → How much pixels vary within each sample")
        
        print(f"\n  Per-sample statistics (mean intensity):")
        print(f"    Mean: {sample_means.mean():.6f}")
        print(f"    Std:  {sample_means.std():.6f}")
        print(f"    Range: [{sample_means.min():.6f}, {sample_means.max():.6f}]")
        
        print(f"\n  Per-sample statistics (std intensity):")
        print(f"    Mean: {sample_stds.mean():.6f}")
        print(f"    Std:  {sample_stds.std():.6f}")
        print(f"    Range: [{sample_stds.min():.6f}, {sample_stds.max():.6f}]")
        
        # Coefficient of Variation (CV) = std / mean
        cv_inter = np.sqrt(inter_sample_var) / heatmap_imp_data.mean() if heatmap_imp_data.mean() != 0 else 0
        cv_intra = np.sqrt(avg_intra_var) / heatmap_imp_data.mean() if heatmap_imp_data.mean() != 0 else 0
        
        print(f"\n  Coefficient of Variation (CV):")
        print(f"    Inter-sample CV: {cv_inter:.4f}")
        print(f"    Intra-sample CV: {cv_intra:.4f}")
        
        # Interpretation
        print(f"\n  *** INTERPRETATION ***")
        if inter_sample_var > 0.01:
            print("  ✅ Good inter-sample variation (samples are diverse)")
        else:
            print("  ⚠️  Low inter-sample variation (samples are too similar)")
        
        if avg_intra_var > 0.01:
            print("  ✅ Good intra-sample variation (spatial patterns exist)")
        else:
            print("  ⚠️  Low intra-sample variation (heatmaps are too uniform)")
    
    # 2. IMPEDANCE VARIATION
    if impedance_vectors:
        print("\n--- IMPEDANCE VARIATION ---")
        impedance_samples = np.array(impedance_vectors)  # Shape: (n_samples, 231)
        
        # Inter-sample variance (variance across samples for each frequency)
        inter_sample_var = np.var(impedance_samples, axis=0).mean()
        
        # Intra-sample variance (variance within each impedance curve)
        intra_sample_vars = [np.var(sample) for sample in impedance_samples]
        avg_intra_var = np.mean(intra_sample_vars)
        
        # Sample statistics
        sample_means = impedance_samples.mean(axis=1)
        sample_stds = impedance_samples.std(axis=1)
        sample_maxs = impedance_samples.max(axis=1)
        
        print(f"  Inter-sample variance: {inter_sample_var:.6f}")
        print(f"    → How much impedance varies across samples at each frequency")
        print(f"  Intra-sample variance (avg): {avg_intra_var:.6f}")
        print(f"    → How much impedance varies within each curve (frequency response)")
        
        print(f"\n  Per-sample statistics (mean impedance):")
        print(f"    Mean: {sample_means.mean():.6f}")
        print(f"    Std:  {sample_means.std():.6f}")
        print(f"    Range: [{sample_means.min():.6f}, {sample_means.max():.6f}]")
        
        print(f"\n  Per-sample statistics (max impedance - peaks):")
        print(f"    Mean: {sample_maxs.mean():.6f}")
        print(f"    Std:  {sample_maxs.std():.6f}")
        print(f"    Range: [{sample_maxs.min():.6f}, {sample_maxs.max():.6f}]")
        
        # Coefficient of Variation
        cv_inter = np.sqrt(inter_sample_var) / impedance_data.mean() if impedance_data.mean() != 0 else 0
        cv_intra = np.sqrt(avg_intra_var) / impedance_data.mean() if impedance_data.mean() != 0 else 0
        
        print(f"\n  Coefficient of Variation (CV):")
        print(f"    Inter-sample CV: {cv_inter:.4f}")
        print(f"    Intra-sample CV: {cv_intra:.4f}")
        
        # Interpretation
        print(f"\n  *** INTERPRETATION ***")
        if inter_sample_var > 0.5:
            print("  ✅ Good inter-sample variation (diverse impedance responses)")
        else:
            print("  ⚠️  Low inter-sample variation (impedance curves too similar)")
        
        if avg_intra_var > 1.0:
            print("  ✅ Good intra-sample variation (rich frequency response)")
        else:
            print("  ⚠️  Low intra-sample variation (flat frequency response)")
        
        if sample_maxs.std() > 1.0:
            print("  ✅ Diverse peak impedance values across samples")
        else:
            print("  ⚠️  Peak impedance values are too similar across samples")
    
    # 3. OCCUPANCY VARIATION
    if occupancy_grids:
        print("\n--- OCCUPANCY VARIATION ---")
        occupancy_samples = np.array(occupancy_grids)  # Shape: (n_samples, 7, 8)
        occupancy_flat = occupancy_samples.reshape(len(occupancy_samples), -1)  # (n_samples, 56)
        
        # Inter-sample variance (how much each cell varies across samples)
        inter_sample_var = np.var(occupancy_flat, axis=0).mean()
        
        # Intra-sample occupancy rate (per sample)
        per_sample_occ_rate = occupancy_flat.mean(axis=1)  # Fraction occupied per sample
        
        # Unique patterns (treat each flattened grid as a binary pattern)
        unique_patterns = len(np.unique(occupancy_flat, axis=0))
        pattern_diversity = unique_patterns / len(occupancy_samples) * 100
        
        print(f"  Inter-sample variance: {inter_sample_var:.6f}")
        print(f"    → How much each cell varies across samples")
        
        print(f"\n  Per-sample occupancy rate:")
        print(f"    Mean: {per_sample_occ_rate.mean():.4f} ({per_sample_occ_rate.mean()*100:.2f}%)")
        print(f"    Std:  {per_sample_occ_rate.std():.4f}")
        print(f"    Range: [{per_sample_occ_rate.min():.4f}, {per_sample_occ_rate.max():.4f}]")
        
        print(f"\n  Pattern diversity:")
        print(f"    Unique patterns: {unique_patterns} out of {len(occupancy_samples)} samples")
        print(f"    Diversity: {pattern_diversity:.2f}%")
        
        # Cell-wise variation (which cells change most across samples)
        cell_vars = np.var(occupancy_flat, axis=0)
        high_var_cells = np.sum(cell_vars > 0.2)  # Cells that vary significantly
        
        print(f"\n  Cell-wise variation:")
        print(f"    Cells with high variation (>0.2): {high_var_cells} / 56")
        print(f"    Average cell variance: {cell_vars.mean():.4f}")
        print(f"    Max cell variance: {cell_vars.max():.4f}")
        
        # Interpretation
        print(f"\n  *** INTERPRETATION ***")
        if pattern_diversity > 50:
            print(f"  ✅ Excellent pattern diversity ({pattern_diversity:.1f}% unique)")
        elif pattern_diversity > 20:
            print(f"  ✓ Good pattern diversity ({pattern_diversity:.1f}% unique)")
        else:
            print(f"  ⚠️  Low pattern diversity ({pattern_diversity:.1f}% unique - many repeated patterns)")
        
        if per_sample_occ_rate.std() > 0.1:
            print("  ✅ Good variation in occupancy rates across samples")
        else:
            print("  ⚠️  Occupancy rates too similar across samples")
        
        if high_var_cells > 28:  # More than half the cells
            print(f"  ✅ Most cells show variation across samples ({high_var_cells}/56)")
        else:
            print(f"  ⚠️  Many cells are static across samples ({high_var_cells}/56 vary)")
    
    # 4. INTER-SAMPLE DIVERSITY (Pairwise Similarity Analysis)
    print("\n" + "="*80)
    print("INTER-SAMPLE DIVERSITY ANALYSIS (How different are samples from each other?)")
    print("="*80)
    
    # HEATMAP Inter-sample Diversity
    if heatmap_impedance_ch:
        print("\n--- HEATMAP INTER-SAMPLE DIVERSITY ---")
        heatmap_samples = np.array(heatmap_impedance_ch)
        heatmap_flat = heatmap_samples.reshape(len(heatmap_samples), -1)
        
        # Sample a subset for pairwise distance calculation (too expensive for 15k samples)
        n_samples_to_compare = min(500, len(heatmap_samples))
        indices = np.random.choice(len(heatmap_samples), n_samples_to_compare, replace=False)
        heatmap_subset = heatmap_flat[indices]
        
        print(f"  Analyzing {n_samples_to_compare} randomly sampled heatmaps...")
        
        # Calculate pairwise Euclidean distances
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(heatmap_subset, metric='euclidean')
        
        # Statistics on pairwise distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        print(f"\n  Pairwise distances (Euclidean):")
        print(f"    Mean: {mean_distance:.4f}")
        print(f"    Std:  {std_distance:.4f}")
        print(f"    Min:  {min_distance:.4f} (most similar pair)")
        print(f"    Max:  {max_distance:.4f} (most different pair)")
        
        # Distance to mean sample
        mean_sample = heatmap_flat.mean(axis=0)
        distances_to_mean = np.sqrt(np.sum((heatmap_flat - mean_sample)**2, axis=1))
        
        print(f"\n  Distance to mean sample:")
        print(f"    Mean: {distances_to_mean.mean():.4f}")
        print(f"    Std:  {distances_to_mean.std():.4f}")
        print(f"    Min:  {distances_to_mean.min():.4f} (most typical sample)")
        print(f"    Max:  {distances_to_mean.max():.4f} (most outlier sample)")
        
        # Nearest neighbor analysis
        dist_matrix = squareform(distances)
        np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distance
        nearest_neighbor_dists = np.min(dist_matrix, axis=1)
        
        print(f"\n  Nearest neighbor distances:")
        print(f"    Mean: {nearest_neighbor_dists.mean():.4f}")
        print(f"    Min:  {nearest_neighbor_dists.min():.4f} (closest pair)")
        print(f"    Max:  {nearest_neighbor_dists.max():.4f} (most isolated sample)")
        
        # Diversity score (normalized standard deviation of distances)
        diversity_score = std_distance / mean_distance if mean_distance > 0 else 0
        
        print(f"\n  Diversity score (CV of distances): {diversity_score:.4f}")
        
        # Interpretation
        print(f"\n  *** INTERPRETATION ***")
        if diversity_score > 0.3:
            print(f"  ✅ High diversity (CV={diversity_score:.3f}) - samples are well-distributed")
        elif diversity_score > 0.15:
            print(f"  ✓ Moderate diversity (CV={diversity_score:.3f}) - reasonable variation")
        else:
            print(f"  ⚠️  Low diversity (CV={diversity_score:.3f}) - samples are too similar")
        
        if nearest_neighbor_dists.min() / mean_distance < 0.1:
            print(f"  ⚠️  Some samples are very similar (close duplicates)")
        else:
            print(f"  ✅ No obvious duplicate samples")
    
    # IMPEDANCE Inter-sample Diversity
    if impedance_vectors:
        print("\n--- IMPEDANCE INTER-SAMPLE DIVERSITY ---")
        impedance_samples = np.array(impedance_vectors)
        
        # Sample subset
        n_samples_to_compare = min(500, len(impedance_samples))
        indices = np.random.choice(len(impedance_samples), n_samples_to_compare, replace=False)
        impedance_subset = impedance_samples[indices]
        
        print(f"  Analyzing {n_samples_to_compare} randomly sampled impedance curves...")
        
        # Pairwise distances
        distances = pdist(impedance_subset, metric='euclidean')
        
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        print(f"\n  Pairwise distances (Euclidean):")
        print(f"    Mean: {mean_distance:.4f}")
        print(f"    Std:  {std_distance:.4f}")
        print(f"    Min:  {min_distance:.4f} (most similar pair)")
        print(f"    Max:  {max_distance:.4f} (most different pair)")
        
        # Distance to mean
        mean_impedance = impedance_samples.mean(axis=0)
        distances_to_mean = np.sqrt(np.sum((impedance_samples - mean_impedance)**2, axis=1))
        
        print(f"\n  Distance to mean impedance curve:")
        print(f"    Mean: {distances_to_mean.mean():.4f}")
        print(f"    Std:  {distances_to_mean.std():.4f}")
        print(f"    Min:  {distances_to_mean.min():.4f}")
        print(f"    Max:  {distances_to_mean.max():.4f}")
        
        # Nearest neighbor
        dist_matrix = squareform(distances)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_neighbor_dists = np.min(dist_matrix, axis=1)
        
        print(f"\n  Nearest neighbor distances:")
        print(f"    Mean: {nearest_neighbor_dists.mean():.4f}")
        print(f"    Min:  {nearest_neighbor_dists.min():.4f}")
        print(f"    Max:  {nearest_neighbor_dists.max():.4f}")
        
        # Diversity score
        diversity_score = std_distance / mean_distance if mean_distance > 0 else 0
        
        print(f"\n  Diversity score (CV of distances): {diversity_score:.4f}")
        
        # Interpretation
        print(f"\n  *** INTERPRETATION ***")
        if diversity_score > 0.3:
            print(f"  ✅ High diversity (CV={diversity_score:.3f}) - impedance curves are varied")
        elif diversity_score > 0.15:
            print(f"  ✓ Moderate diversity (CV={diversity_score:.3f})")
        else:
            print(f"  ⚠️  Low diversity (CV={diversity_score:.3f}) - curves are similar")
    
    # OCCUPANCY Inter-sample Diversity
    if occupancy_grids:
        print("\n--- OCCUPANCY INTER-SAMPLE DIVERSITY ---")
        occupancy_samples = np.array(occupancy_grids)
        occupancy_flat = occupancy_samples.reshape(len(occupancy_samples), -1)
        
        # Sample subset
        n_samples_to_compare = min(500, len(occupancy_samples))
        indices = np.random.choice(len(occupancy_samples), n_samples_to_compare, replace=False)
        occupancy_subset = occupancy_flat[indices]
        
        print(f"  Analyzing {n_samples_to_compare} randomly sampled occupancy grids...")
        
        # For binary data, use Hamming distance (fraction of differing bits)
        distances = pdist(occupancy_subset, metric='hamming') * 56  # Multiply by grid size for absolute count
        
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        print(f"\n  Pairwise distances (Hamming - differing cells):")
        print(f"    Mean: {mean_distance:.2f} cells differ")
        print(f"    Std:  {std_distance:.2f}")
        print(f"    Min:  {min_distance:.2f} (most similar pair)")
        print(f"    Max:  {max_distance:.2f} (most different pair)")
        
        # Distance to mean (most common pattern at each cell)
        mean_occupancy = (occupancy_flat.mean(axis=0) > 0.5).astype(float)
        distances_to_mean = np.sum(occupancy_flat != mean_occupancy, axis=1)
        
        print(f"\n  Distance to most common pattern:")
        print(f"    Mean: {distances_to_mean.mean():.2f} cells differ")
        print(f"    Std:  {distances_to_mean.std():.2f}")
        print(f"    Min:  {distances_to_mean.min():.0f}")
        print(f"    Max:  {distances_to_mean.max():.0f}")
        
        # Nearest neighbor
        dist_matrix = squareform(distances)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_neighbor_dists = np.min(dist_matrix, axis=1)
        
        print(f"\n  Nearest neighbor distances:")
        print(f"    Mean: {nearest_neighbor_dists.mean():.2f} cells differ")
        print(f"    Min:  {nearest_neighbor_dists.min():.2f}")
        print(f"    Max:  {nearest_neighbor_dists.max():.2f}")
        
        # Diversity score
        diversity_score = std_distance / mean_distance if mean_distance > 0 else 0
        
        print(f"\n  Diversity score (CV of distances): {diversity_score:.4f}")
        
        # Interpretation
        print(f"\n  *** INTERPRETATION ***")
        if mean_distance > 20:
            print(f"  ✅ High diversity - average {mean_distance:.1f}/56 cells differ between samples")
        elif mean_distance > 10:
            print(f"  ✓ Moderate diversity - average {mean_distance:.1f}/56 cells differ")
        else:
            print(f"  ⚠️  Low diversity - only {mean_distance:.1f}/56 cells differ on average")
        
        if nearest_neighbor_dists.min() < 3:
            print(f"  ⚠️  Some samples differ by only {nearest_neighbor_dists.min():.0f} cells (near duplicates)")
        else:
            print(f"  ✅ No near-duplicate samples (min difference: {nearest_neighbor_dists.min():.0f} cells)")

    # 5. OVERALL DATASET QUALITY
    print("\n" + "="*80)
    print("OVERALL DATASET QUALITY FOR LEARNING")
    print("="*80)
    
    quality_score = 0
    max_score = 0
    
    if heatmap_impedance_ch:
        max_score += 3
        if inter_sample_var > 0.01:
            quality_score += 1
            print("✅ Heatmap: Diverse across samples")
        else:
            print("⚠️  Heatmap: Low diversity across samples")
        
        if avg_intra_var > 0.01:
            quality_score += 1
            print("✅ Heatmap: Rich spatial patterns")
        else:
            print("⚠️  Heatmap: Weak spatial patterns")
        
        if sample_means.std() > 0.05:
            quality_score += 1
            print("✅ Heatmap: Good range of intensities")
        else:
            print("⚠️  Heatmap: Limited intensity range")
    
    if impedance_vectors:
        max_score += 3
        impedance_samples = np.array(impedance_vectors)
        inter_sample_var_imp = np.var(impedance_samples, axis=0).mean()
        sample_maxs_imp = impedance_samples.max(axis=1)
        
        if inter_sample_var_imp > 0.5:
            quality_score += 1
            print("✅ Impedance: Diverse frequency responses")
        else:
            print("⚠️  Impedance: Similar frequency responses")
        
        if np.mean([np.var(s) for s in impedance_samples]) > 1.0:
            quality_score += 1
            print("✅ Impedance: Rich spectral features")
        else:
            print("⚠️  Impedance: Flat spectral features")
        
        if sample_maxs_imp.std() > 1.0:
            quality_score += 1
            print("✅ Impedance: Diverse peak values")
        else:
            print("⚠️  Impedance: Similar peak values")
    
    if occupancy_grids:
        max_score += 2
        occupancy_flat_check = np.array(occupancy_grids).reshape(len(occupancy_grids), -1)
        unique_patterns_check = len(np.unique(occupancy_flat_check, axis=0))
        pattern_diversity_check = unique_patterns_check / len(occupancy_samples) * 100
        
        if pattern_diversity_check > 20:
            quality_score += 1
            print("✅ Occupancy: Good pattern diversity")
        else:
            print("⚠️  Occupancy: Low pattern diversity")
        
        if np.var(occupancy_flat_check, axis=0).mean() > 0.15:
            quality_score += 1
            print("✅ Occupancy: Cells vary across samples")
        else:
            print("⚠️  Occupancy: Cells too static")
    
    print(f"\n{'='*80}")
    print(f"DATASET QUALITY SCORE: {quality_score}/{max_score}")
    print(f"{'='*80}")
    
    if quality_score >= max_score * 0.8:
        print("✅ EXCELLENT: Dataset has strong variation - model can learn rich patterns")
    elif quality_score >= max_score * 0.6:
        print("✓ GOOD: Dataset has adequate variation - model should learn well")
    elif quality_score >= max_score * 0.4:
        print("⚠️  FAIR: Dataset has limited variation - model may struggle with generalization")
    else:
        print("❌ POOR: Dataset lacks variation - model will likely overfit or underperform")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total samples analyzed: {n_samples}")
    print(f"Heatmaps: {len(heatmap_impedance_ch)}")
    print(f"Impedances: {len(impedance_vectors)}")
    print(f"Occupancy grids: {len(occupancy_grids)}")
    print("="*80 + "\n")


def quick_stats_normalized(data_root="datasets/data_norm"):
    """Quick statistics for normalized dataset: mean, median, skewness."""
    data_root = Path(data_root)
    
    print("\n" + "="*80)
    print("NORMALIZED DATASET QUICK STATISTICS")
    print("="*80)
    print(f"Data directory: {data_root}\n")
    
    # Heatmap
    heatmap_dir = data_root / "heatmap"
    if heatmap_dir.exists():
        heatmap_files = sorted(heatmap_dir.glob("sample_*.npy"))[:100]  # Sample 100 files
        heatmap_ch0_values = []
        for f in heatmap_files:
            data = np.load(f)
            if data.shape[0] >= 1:
                heatmap_ch0_values.append(data[0].flatten())
        
        if heatmap_ch0_values:
            heatmap_data = np.concatenate(heatmap_ch0_values)
            print("HEATMAP (normalized):")
            print(f"  Mean:     {heatmap_data.mean():.6f}")
            print(f"  Median:   {np.median(heatmap_data):.6f}")
            print(f"  Skewness: {stats.skew(heatmap_data):.6f}")
    
    # Impedance
    imp_dir = data_root / "Imp"
    if imp_dir.exists():
        imp_files = sorted(imp_dir.glob("sample_*.npy"))[:100]
        imp_values = np.concatenate([np.load(f).flatten() for f in imp_files])
        print("\nIMPEDANCE (normalized):")
        print(f"  Mean:     {imp_values.mean():.6f}")
        print(f"  Median:   {np.median(imp_values):.6f}")
        print(f"  Skewness: {stats.skew(imp_values):.6f}")
    
    # Max Value
    max_val_dir = data_root / "Max_value"
    if max_val_dir.exists():
        max_val_files = sorted(max_val_dir.glob("sample_*.npy"))[:100]
        max_values = np.concatenate([np.load(f).flatten() for f in max_val_files])
        print("\nMAX_VALUE (normalized):")
        print(f"  Mean:     {max_values.mean():.6f}")
        print(f"  Median:   {np.median(max_values):.6f}")
        print(f"  Skewness: {stats.skew(max_values):.6f}")
    
    # Occupancy
    occ_dir = data_root / "Occ_map"
    if occ_dir.exists():
        occ_files = sorted(occ_dir.glob("sample_*.npy"))[:100]
        occ_values = np.concatenate([np.load(f).flatten() for f in occ_files])
        print("\nOCCUPANCY (binary):")
        print(f"  Mean:     {occ_values.mean():.6f}")
        print(f"  Median:   {np.median(occ_values):.6f}")
        print(f"  Skewness: {stats.skew(occ_values):.6f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "norm":
        # Quick stats for normalized data
        quick_stats_normalized(data_root="datasets/data_norm")
    else:
        # Full statistics for raw data (default)
        calculate_statistics(data_root="datasets/data")