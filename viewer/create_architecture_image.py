"""
Create VAE Architecture Diagram as PNG image using matplotlib
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(8, 11.5, 'VAE Multi-Input Architecture with Attention Mechanisms', 
        ha='center', va='center', fontsize=18, fontweight='bold')

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, color='lightblue', fontsize=9):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', wrap=True)

def create_arrow(ax, x1, y1, x2, y2, style='->', color='black', linewidth=2.0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle=style, color=color, linewidth=linewidth,
                           mutation_scale=20)
    ax.add_patch(arrow)

# ENCODER SECTION
# Heatmap Branch
create_box(ax, 0.5, 9, 1.2, 0.7, 'Heatmap\n64×64×2', '#FFE5E5', 8)
create_box(ax, 0.5, 7.5, 1.2, 0.7, 'Conv\nBlocks', '#FFB3B3', 8)
create_box(ax, 0.5, 6, 1.2, 0.7, '4×4×256', '#FF8080', 8)
create_box(ax, 0.5, 4.5, 1.2, 0.7, '🔍 Self-Attn\n2D', '#FF4444', 7)
create_box(ax, 0.5, 3, 1.2, 0.7, 'FC\nμ_h, σ_h', '#FFB3B3', 8)

create_arrow(ax, 1.1, 8.85, 1.1, 8.2)
create_arrow(ax, 1.1, 7.35, 1.1, 6.7)
create_arrow(ax, 1.1, 5.85, 1.1, 5.2)
create_arrow(ax, 1.1, 4.35, 1.1, 3.7)

# Occupancy Branch
create_box(ax, 2.3, 9, 1.2, 0.7, 'Occupancy\n7×8×1', '#E5F5E5', 8)
create_box(ax, 2.3, 7.5, 1.2, 0.7, 'Conv\nBlocks', '#B3E5B3', 8)
create_box(ax, 2.3, 6, 1.2, 0.7, '7×8×32', '#80D580', 8)
create_box(ax, 2.3, 4.5, 1.2, 0.7, '🔍 Self-Attn\n2D', '#44CC44', 7)
create_box(ax, 2.3, 3, 1.2, 0.7, 'FC\nμ_o, σ_o', '#B3E5B3', 8)

create_arrow(ax, 2.9, 8.85, 2.9, 8.2)
create_arrow(ax, 2.9, 7.35, 2.9, 6.7)
create_arrow(ax, 2.9, 5.85, 2.9, 5.2)
create_arrow(ax, 2.9, 4.35, 2.9, 3.7)

# Impedance Branch
create_box(ax, 4.1, 9, 1.2, 0.7, 'Impedance\n231×1', '#FFF5E5', 8)
create_box(ax, 4.1, 7.5, 1.2, 0.7, 'MLP\nLayers', '#FFE5B3', 8)
create_box(ax, 4.1, 6, 1.2, 0.7, '128-dim', '#FFD580', 8)
create_box(ax, 4.1, 4.5, 1.2, 0.7, '🔍 Self-Attn\n1D', '#FFBB44', 7)
create_box(ax, 4.1, 3, 1.2, 0.7, 'FC\nμ_i, σ_i', '#FFE5B3', 8)

create_arrow(ax, 4.7, 8.85, 4.7, 8.2)
create_arrow(ax, 4.7, 7.35, 4.7, 6.7)
create_arrow(ax, 4.7, 5.85, 4.7, 5.2)
create_arrow(ax, 4.7, 4.35, 4.7, 3.7)

# FUSION SECTION
create_box(ax, 0.5, 1.2, 4.8, 1, 'Product of Experts (PoE)\nPrecision-Weighted Fusion', '#CCFFCC', 9)
create_arrow(ax, 1.1, 3, 2.9, 2.2)
create_arrow(ax, 2.9, 3, 2.9, 2.2)
create_arrow(ax, 4.7, 3, 2.9, 2.2)

create_box(ax, 1.5, 0.2, 2, 0.7, 'z (latent)', '#AAFFAA', 9)
create_arrow(ax, 2.9, 1.2, 2.5, 0.9)

# DECODER SECTION
# Impedance Decoder
create_box(ax, 6.5, 0.2, 1.8, 0.7, 'Impedance\nDecoder', '#FFE5B3', 8)
create_arrow(ax, 3.5, 0.55, 6.5, 0.55)
create_box(ax, 8.8, 0.2, 1.3, 0.7, 'Impedance\n231×1', '#FFF5E5', 8)
create_arrow(ax, 8.3, 0.55, 8.8, 0.55)

# Project for cross-attention
create_box(ax, 10.5, 0.2, 1.5, 0.7, 'Project to\nlatent_dim', '#FFCC99', 7)
create_arrow(ax, 10.1, 0.55, 10.5, 0.55)

# Heatmap Decoder
create_box(ax, 6.5, 2, 1.3, 0.7, 'FC &\nReshape', '#FFB3B3', 8)
create_arrow(ax, 3.5, 0.55, 6.5, 2.35)

create_box(ax, 6.5, 3.2, 1.3, 0.7, 'Upsample\n8×8', '#FF8080', 8)
create_arrow(ax, 7.15, 2.7, 7.15, 3.2)

create_box(ax, 6.5, 4.4, 1.3, 0.7, 'Upsample\n16×16', '#FF8080', 8)
create_arrow(ax, 7.15, 3.9, 7.15, 4.4)

create_box(ax, 6.5, 5.6, 1.5, 0.7, '🔍 Self-Attn\n2D Blueprint', '#FF4444', 7)
create_arrow(ax, 7.15, 5.1, 7.15, 5.6)

create_box(ax, 6.5, 6.8, 1.8, 0.7, '⚡ Cross-Attn\nHeatmap←Imp', '#66CCFF', 7)
create_arrow(ax, 7.25, 6.3, 7.25, 6.8)

# Cross-attention connection from impedance
create_arrow(ax, 11.25, 0.9, 11.25, 5.5, color='blue', linewidth=1.5)
create_arrow(ax, 11.25, 5.5, 8.3, 7.15, color='blue', linewidth=1.5)

create_box(ax, 6.5, 8, 1.3, 0.7, 'Upsample\n64×64', '#FF8080', 8)
create_arrow(ax, 7.25, 7.5, 7.25, 8)

create_box(ax, 6.5, 9, 1.3, 0.7, 'Heatmap\n64×64×2', '#FFE5E5', 8)
create_arrow(ax, 7.15, 8.7, 7.15, 9)

# Occupancy Decoder
create_box(ax, 9.5, 2, 1.5, 0.7, 'Occupancy\nDecoder', '#B3E5B3', 8)
create_arrow(ax, 3.5, 0.55, 9.5, 2.35)

create_box(ax, 9.5, 3.2, 1.5, 0.7, 'Occupancy\n7×8×1', '#E5F5E5', 8)
create_arrow(ax, 10.25, 2.7, 10.25, 3.2)

# Add legend
legend_y = 10.5
ax.text(12, legend_y, 'Legend:', fontsize=10, fontweight='bold')
legend_elements = [
    mpatches.Patch(color='#FF4444', label='Self-Attention 2D'),
    mpatches.Patch(color='#FFBB44', label='Self-Attention 1D'),
    mpatches.Patch(color='#66CCFF', label='Cross-Attention'),
    mpatches.Patch(color='#CCFFCC', label='PoE Fusion'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
          bbox_to_anchor=(0.99, 0.92))

# Add section labels
ax.text(2.5, 10.3, 'ENCODER', fontsize=12, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(2.5, 1.8, 'FUSION', fontsize=12, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax.text(8.5, 10.3, 'DECODER', fontsize=12, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Tight layout and save
plt.tight_layout()
plt.savefig('temp_visuals/vae_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Architecture diagram saved as: temp_visuals/vae_architecture.png")
print("📊 Image size: 4800×3600 pixels (300 DPI)")
