import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless server
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path
import json

#=========================================================================
# Script to verify normalization of heatmap, impedance, and occupancy data
# - Heatmap: log(1+x) z-score on foreground pixels, sentinel for background
#   Background pixels shown in WHITE to distinguish from data
# - Impedance: log z-score normalization
# - Occupancy: Binary data (copied as-is)
# - Plots normalized and denormalized data side by side
#=========================================================================

repo_root = Path(__file__).resolve().parents[1]
output_dir = repo_root / "temp_visuals"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load normalization stats ---
norm_root = repo_root / "datasets" / "data_norm"
stats_file = norm_root / "normalization_stats.json"
with open(stats_file, "r") as f:
    stats = json.load(f)

SENTINEL = stats["background_value"]          # -3.6228
log_mean = stats["Heatmap"]["log_mean"]        # 0.3944
log_std  = stats["Heatmap"]["log_std"]         # 0.1789
imp_log_mean = stats["Impedance"]["log_mean"]  # -1.1628
imp_log_std  = stats["Impedance"]["log_std"]   # 1.8059

print(f"Sentinel background value: {SENTINEL}")
print(f"Heatmap  log(1+x) z-score: mean={log_mean:.4f}, std={log_std:.4f}")
print(f"Impedance log z-score:     mean={imp_log_mean:.4f}, std={imp_log_std:.4f}")


# --- Denormalization helpers ---
def denorm_heatmap(z_data):
    """z-score → log(1+x) → raw.  Returns (raw, bg_mask)."""
    bg_mask = np.isclose(z_data, SENTINEL, atol=0.01)
    log1p_vals = z_data * log_std + log_mean          # undo z-score
    raw = np.expm1(np.clip(log1p_vals, 0, None))      # exp(v)-1, clamp ≥ 0
    raw[bg_mask] = np.nan                              # NaN → white in imshow
    return raw, bg_mask


def denorm_impedance(z_data):
    """z-score → log → raw impedance."""
    log_vals = z_data * imp_log_std + imp_log_mean
    return np.exp(log_vals)


# --- Load sample data ---
heatmap_dir = norm_root / "heatmap"
imp_dir     = norm_root / "Imp"
occ_dir     = norm_root / "Occ_map"
freq_file   = repo_root / "configs" / "Frequency_data_hz.npy"

heatmap_files = sorted(heatmap_dir.glob("*.npy"))
imp_files     = sorted(imp_dir.glob("*.npy"))
occ_files     = sorted(occ_dir.glob("*.npy"))

SAMPLE_IDX = 2  # change to inspect a different sample
hm_norm  = np.load(heatmap_files[SAMPLE_IDX]).astype(np.float32)   # (1,64,64)
imp_norm = np.load(imp_files[SAMPLE_IDX]).astype(np.float32)       # (231,) or (231,1)
occ      = np.load(occ_files[SAMPLE_IDX]).astype(np.float32)       # (52,)
freq     = np.load(freq_file).flatten() if freq_file.exists() else np.arange(imp_norm.size)

print(f"\nSample: {heatmap_files[SAMPLE_IDX].name}")
print(f"  Heatmap  shape={hm_norm.shape}  range=[{hm_norm.min():.4f}, {hm_norm.max():.4f}]")
print(f"  Impedance shape={imp_norm.shape}  range=[{imp_norm.min():.4f}, {imp_norm.max():.4f}]")
print(f"  Occupancy shape={occ.shape}  unique={np.unique(occ)}")

# Handle both old (231,) single-channel and new (2,231) dual-channel impedance
if imp_norm.ndim == 2 and imp_norm.shape[0] == 2:
    imp_zscore = imp_norm[0]   # Ch0: raw z-score
    imp_deriv  = imp_norm[1]   # Ch1: first derivative of z-score
    print(f"  Dual-channel impedance: Ch0=z-score, Ch1=derivative")
else:
    imp_zscore = imp_norm.flatten()
    imp_deriv  = None
    print(f"  Single-channel impedance")

# Denormalize
hm_raw, bg_mask = denorm_heatmap(hm_norm[0])  # (64,64)
imp_raw = denorm_impedance(imp_zscore)

# Count background / foreground
n_bg = bg_mask.sum()
n_fg = (~bg_mask).sum()
print(f"  Background pixels: {n_bg}/{hm_norm[0].size}  ({100*n_bg/hm_norm[0].size:.1f}%)")

# --- Build custom colormap: jet with white for NaN (background) ---
jet = cm.get_cmap("jet", 256)
jet_white = jet.copy()
jet_white.set_bad(color="white")  # NaN → white


# ---------- FIGURE ----------
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle(f"Normalization Verification — {heatmap_files[SAMPLE_IDX].stem}", fontsize=14, y=0.98)

# ── Row 0: Heatmap ─────────────────────────────────────────────────────
# Col 0: Normalized z-score (background = sentinel, shown in white)
hm_display = hm_norm[0].copy()
hm_display[bg_mask] = np.nan
im0 = axes[0, 0].imshow(hm_display, cmap=jet_white)
axes[0, 0].set_title(f"Heatmap Normalized (z-score)\nbg sentinel={SENTINEL:.2f} → white")
fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

# Col 1: Denormalized (physical units), background = white
im1 = axes[0, 1].imshow(hm_raw, cmap=jet_white)
axes[0, 1].set_title("Heatmap Denormalized (physical)")
fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Col 2: Background mask (white = background, black = foreground)
axes[0, 2].imshow(bg_mask.astype(float), cmap="gray_r", vmin=0, vmax=1)
axes[0, 2].set_title(f"Background Mask\nwhite=bg ({n_bg}px), black=fg ({n_fg}px)")

# ── Row 1: Impedance ──────────────────────────────────────────────────
# Col 0: Normalized z-score Ch0 (linear y, log x)
axes[1, 0].plot(freq, imp_zscore, "b-", linewidth=1, label="Ch0 z-score")
axes[1, 0].set_xscale("log")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("z-score")
axes[1, 0].set_title("Impedance Normalized — Ch0 (z-score)")
axes[1, 0].grid(True, alpha=0.3)

# Col 1: Derivative Ch1 (or denorm physical if single-channel)
if imp_deriv is not None:
    axes[1, 1].plot(freq, imp_deriv, color="darkorange", linewidth=1, label="Ch1 derivative")
    axes[1, 1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("First Difference (Δz)")
    axes[1, 1].set_title("Impedance — Ch1 (First Derivative of z-score)")
    axes[1, 1].grid(True, alpha=0.3)
else:
    axes[1, 1].loglog(freq, imp_raw, "g-", linewidth=1)
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Impedance (Ω)")
    axes[1, 1].set_title("Impedance Denormalized (physical)")
    axes[1, 1].grid(True, alpha=0.3, which="both")

# Col 2: Denormalized physical + derivative overlaid on twin axis
axes[1, 2].loglog(freq, imp_raw, "b-", linewidth=1.5, label="Denorm (Ω)")
axes[1, 2].set_xlabel("Frequency (Hz)")
axes[1, 2].set_ylabel("Impedance (Ω)", color="b")
axes[1, 2].tick_params(axis="y", labelcolor="b")
title_str = "Impedance Denormalized"
if imp_deriv is not None:
    ax_d = axes[1, 2].twinx()
    ax_d.plot(freq, imp_deriv, color="darkorange", linewidth=1,
              linestyle=":", label="Ch1 derivative")
    ax_d.axhline(0, color="darkorange", linewidth=0.5, linestyle=":", alpha=0.5)
    ax_d.set_ylabel("First Derivative (Δz)", color="darkorange")
    ax_d.tick_params(axis="y", labelcolor="darkorange")
    lines1, labels1 = axes[1, 2].get_legend_handles_labels()
    lines2, labels2 = ax_d.get_legend_handles_labels()
    axes[1, 2].legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    title_str += " + Derivative"
else:
    axes[1, 2].legend(loc="best", fontsize=8)
axes[1, 2].set_title(title_str)
axes[1, 2].grid(True, alpha=0.3, which="both")

# ── Row 2: Occupancy ──────────────────────────────────────────────────
# Col 0: Bar chart of occupancy vector
axes[2, 0].bar(range(len(occ)), occ, color="steelblue", edgecolor="k", linewidth=0.3)
axes[2, 0].set_xlabel("Region Index")
axes[2, 0].set_ylabel("Occupied (0/1)")
axes[2, 0].set_title(f"Occupancy Vector ({int(occ.sum())}/{len(occ)} active)")
axes[2, 0].set_ylim(-0.05, 1.05)

# Col 1: z-score histogram of foreground heatmap pixels
fg_vals = hm_norm[0][~bg_mask]
axes[2, 1].hist(fg_vals, bins=80, color="coral", edgecolor="k", linewidth=0.3, density=True)
axes[2, 1].axvline(0, color="k", linestyle="--", linewidth=0.8, label="z=0")
axes[2, 1].set_xlabel("z-score")
axes[2, 1].set_ylabel("Density")
axes[2, 1].set_title(f"Heatmap Foreground z-score Distribution\n"
                      f"min={fg_vals.min():.2f}, max={fg_vals.max():.2f}")
axes[2, 1].legend()

# Col 2: impedance z-score histogram
axes[2, 2].hist(imp_zscore.flatten(), bins=60, color="skyblue", edgecolor="k", linewidth=0.3, density=True)
axes[2, 2].axvline(0, color="k", linestyle="--", linewidth=0.8, label="z=0")
axes[2, 2].set_xlabel("z-score")
axes[2, 2].set_ylabel("Density")
axes[2, 2].set_title(f"Impedance z-score Distribution (Ch0)\n"
                      f"min={imp_zscore.min():.2f}, max={imp_zscore.max():.2f}")
axes[2, 2].legend()

plt.tight_layout()
out_path = output_dir / "normalization_verification.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved to {out_path}")
plt.close(fig)

