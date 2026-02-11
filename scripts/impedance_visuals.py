import numpy as np
import matplotlib.pyplot as plt


def Impedance_profile(real_magnitude, generated_impedance, output_path):
   
    target_impedance=np.load('configs/target_impedance.npy'),
    frequency=np.load('configs/Frequency_data_hz.npy'),
   # Plot the Impedance profiles
    plt.figure(figsize=(15, 9))
    # Plot generated impedance

    # Plot real magnitude
    #plt.loglog(frequency, real_magnitude, marker='', markersize=5, linestyle='-', linewidth=5,  color='green',label='Real Magnitude')
    #Plot target impedance
    plt.loglog(frequency, target_impedance, marker='',markersize=2, linestyle='--', linewidth=3.5, label='Target Impedance (TI)', color='red')
    if generated_impedance is not None:
        plt.loglog(
            frequency,
            generated_impedance,
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


if __name__ == "__main__":
    # Load the real and generated impedance data
    #real_impedances = np.load('data/real_impedances.npy')  # Shape: (num_samples, num_points)
    #fake_impedances = np.load('data/fake_impedances.npy')  # Shape: (num_samples, num_points)
    frequency = np.load('configs/Frequency_data_hz.npy')  # Shape: (num_points,)
    target_impedance = np.load('configs/target_impedance.npy')  # Shape: (num_points, 1)
    #real_magnitude = real_impedances.flatten()
    #generated_impedance = fake_impedances.flatten()

    Impedance_profile(
        real_magnitude=None,
        generated_impedance=None,
        output_path='temp_visuals/generated_vs_target_impedance_npy.png',
    )
