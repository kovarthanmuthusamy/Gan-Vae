import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.path import Path
from scipy.interpolate import griddata, RBFInterpolator
import os


def plot_impedance_heatmap_clean(file_path):

    # -------------------------------------------------
    # 1. LOAD MAP DATA
    # -------------------------------------------------
    x, y, z = [], [], []

    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip() == "3":
            try:
                for k in range(1, 4):
                    px, py, pz = map(float, lines[i + k].split())
                    x.append(px)
                    y.append(py)
                    z.append(max(pz, 0.01))  # avoid zeros
                i += 3
            except:
                pass
        i += 1

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    if len(x) == 0:
        raise RuntimeError("No data loaded from MAP file.")

    # -------------------------------------------------
    # 2. SCALE TO mm
    # -------------------------------------------------
    scale = 1e-5  # 10 nm → mm
    x = (x - x.min()) * scale
    y = (y - y.min()) * scale

    print(f"Loaded {len(x)} points")
    print(f"Board size ≈ {x.max():.1f} × {y.max():.1f} mm")

    # -------------------------------------------------
    # 3. REGULAR GRID
    # -------------------------------------------------
    resolution = 64
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    # -------------------------------------------------
    # 4. EXACT H-SHAPE MASK (FIXED GEOMETRY)
    # -------------------------------------------------
    # !!! Adjust vertices ONCE to match your board !!!
    H_vertices = [
        (0, 0), (89, 0), (89, 56), (209, 56),
        (209, 0), (300, 0), (300, 320),
        (209, 320), (209, 258), (89, 258),
        (89, 320), (0, 320)
    ]

    board_path = Path(H_vertices)

    mask_board = board_path.contains_points(
        np.column_stack((Xi.ravel(), Yi.ravel()))
    ).reshape(Xi.shape)

    np.save("binary_mask.npy", mask_board)

    # -------------------------------------------------
    # 5. FIELD INTERPOLATION (SMOOTH & PHYSICAL)
    # -------------------------------------------------
    points = np.column_stack((x, y))

    # --- Option A: Linear (faster, needs denser points)
    Zi = griddata(points, z, (Xi, Yi), method="linear")

    # --- Option B: RBF (recommended – closer to EM solver)
    nan_mask = np.isnan(Zi)
    if np.any(nan_mask):
        rbf = RBFInterpolator(points, z, smoothing=0.15)
        Zi[nan_mask] = rbf(
            np.column_stack((Xi[nan_mask], Yi[nan_mask]))
        )

    Zi = np.ma.masked_where(~mask_board, Zi)

    # -------------------------------------------------
    # 6. PLOTTING
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 7))

    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")

    # --- Automatic color limits from data ---
    Zi_valid = Zi.compressed()  # removes masked values

    vmin = Zi_valid.min()
    vmax = Zi_valid.max()

    norm = colors.PowerNorm(gamma=2.2, vmin=vmin, vmax=vmax)

    img = ax.imshow(
        Zi,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        aspect="equal"
    )

    ax.set_title("Impedance Distribution @ 63 MHz", fontsize=14)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, linestyle=":", alpha=0.3)

    cbar = plt.colorbar(img, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Impedance (Ohm)", rotation=270, labelpad=18)

    plt.tight_layout()
    plt.savefig("temp_visuals/impedance_clean.png", dpi=300)
    plt.show()


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
    img = ax.imshow(
        heatmap,
        cmap="jet",
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
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    file_name = "scripts/Z_0063.000MHz.map"
    if os.path.exists(file_name):
        plot_impedance_heatmap_clean(file_name)
    else:
        print("MAP file not found.")
