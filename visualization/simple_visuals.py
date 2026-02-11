import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from pathlib import Path

### helper function to plot heatmap arrays (from visualization_utils.py)
def plot_heatmap_array(
    heatmap,
    output_path,
    title="Generated Heatmap",
    mask=None,
    threshold=0.5,
    vmin=None,
    vmax=None,
):
    heatmap = np.asarray(heatmap)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap array, got shape {heatmap.shape}")

    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != heatmap.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match heatmap {heatmap.shape}")
        heatmap = np.ma.masked_where(mask < threshold, heatmap)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap_22 = mpl.colormaps['jet'].resampled(22)
    img = ax.imshow(
        heatmap,
        cmap=cmap_22,
       # origin="lower",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300,bbox_inches='tight')
    plt.close(fig)


# Impedance denormalization function
def denormalize_impedance(normalized_imp, log_min, log_max):
    """
    Denormalize log-scale normalized impedance data
    
    Args:
        normalized_imp: Normalized impedance in [0, 1] range
        log_min: Minimum value in log-space used during normalization
        log_max: Maximum value in log-space used during normalization
    
    Returns:
        Denormalized impedance in original Ohm scale
    """
    # Step 1: Scale from [0, 1] back to log-space
    log_data = normalized_imp * (log_max - log_min) + log_min
    # Step 2: Exponentiate to get original scale
    return np.exp(log_data)


# Impedance profile plotting function (from impedance_visuals.py)
def Impedance_profile(generated_impedance, output_path,stats_path,stats_type="percentile_min_max"):
    """
    Plot impedance profile comparison with automatic denormalization
    
    Args:
        generated_impedance: Normalized impedance from model output [0, 1] range
        output_path: Path to save the plot
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    percentile_stats = stats.get(stats_type, {})
    log_min = percentile_stats.get("imp_log_min")
    log_max = percentile_stats.get("imp_log_max")
    
    target_impedance = np.load('configs/target_impedance.npy').squeeze()
    frequency = np.load('configs/Frequency_data_hz.npy').squeeze()
    
    # Plot the Impedance profiles
    plt.figure(figsize=(15, 9))
    
    # Plot target impedance (already in Ohm scale)
    plt.loglog(frequency, target_impedance, marker='', markersize=2, linestyle='--', 
               linewidth=3.5, label='Target Impedance (TI)', color='red')
    
    # Plot generated impedance with denormalization
    if generated_impedance is not None:
        generated_impedance = np.asarray(generated_impedance).squeeze()
        
        # Denormalize from [0, 1] to original Ohm scale
        generated_impedance_denorm = denormalize_impedance(generated_impedance, log_min, log_max)
        
        plt.loglog(
            frequency,
            generated_impedance_denorm,  # Use denormalized values
            marker='',
            markersize=2,
            linestyle='-',
            linewidth=3.5,
            label='Generated Impedance',
            color='blue',
        )
    plt.ylim(1e-3, 1e2)     # Impedance range (Ohm)
    #x-axis-title
    plt.xlabel("Frequency (Hz)")
    #y-axis-title
    plt.ylabel("Impedance (Ohm)")
    #plot-title
    plt.title("Impedance Profile Comparison")
    plt.legend(fontsize=22)
    plt.grid(True, which="both")
    # for fitting the layout
    plt.tight_layout()
    # save the plot 
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")
    plt.clf()



heatmap = np.load("experiments/exp009/visuals/data_sample_0/heatmap.npy")
Imp = np.load("experiments/exp009/visuals/data_sample_0/impedance_profile.npy")
stats_path = Path("datasets/source/data_norm/normalization_stats.json")
heatmap_ch0 = heatmap[0]  # Extract channel 0 (impedance channel)
out_path = "temp_visuals/heatmap_sample.png"
out_Im_path = out_path.replace("heatmap_sample", "impedance_profile")

plot_heatmap_array(heatmap=heatmap_ch0, output_path=out_path, vmin=0.0, vmax=1.0)
Impedance_profile(generated_impedance=Imp, output_path=out_Im_path, stats_path=stats_path)