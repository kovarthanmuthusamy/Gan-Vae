# Visualization Guide

## Overview
Visualization functions have been added to all three modality modules: heatmap, impedance, and occupancy grid.

## Module Visualization Functions

### 1. **heatmap.py** - `visualize_heatmap()`
Visualizes 2-channel heatmap data:
- Channel 1: Impedance heatmap (hot colormap)
- Channel 2: Board mask (binary colormap)

**Usage:**
```python
from heatmap import visualize_heatmap

# Visualize and display
visualize_heatmap('sample_1.npy')

# Save without displaying
visualize_heatmap('sample_1.npy', output_path='heatmap_viz.png', show=False)
```

**Parameters:**
- `heatmap_file`: Path to .npy file (shape: 2, 64, 64)
- `output_path`: Save location (optional)
- `show`: Display plot (default: True)

---

### 2. **impedance.py** - `visualize_impedance()`
Visualizes impedance profile with log-log plot:
- Generated impedance (blue line)
- Target impedance (red dashed line, if available)
- Frequency range on x-axis
- Impedance range on y-axis

**Usage:**
```python
from impedance import visualize_impedance

# Visualize and display
visualize_impedance('sample_1.npy')

# Save without displaying
visualize_impedance('sample_1.npy', output_path='impedance_viz.png', show=False)
```

**Parameters:**
- `impedance_file`: Path to .npy file (shape: 231, 1)
- `output_path`: Save location (optional)
- `show`: Display plot (default: True)

**Requirements:**
- `configs/Frequency_data_hz.npy`
- `configs/target_impedance.npy`

---

### 3. **occupancy.py** - `visualize_occupancy_grid()`
Visualizes occupancy grid with cell annotations:
- Grid size: 7×8
- Color: Red (active=1), Yellow (inactive=0)
- Cell values displayed as text
- Grid lines for clarity

**Usage:**
```python
from occupancy import visualize_occupancy_grid

# Visualize and display
visualize_occupancy_grid('sample_1.npy')

# Save without displaying
visualize_occupancy_grid('sample_1.npy', output_path='occupancy_viz.png', show=False)
```

**Parameters:**
- `occ_grid_file`: Path to .npy file (shape: 7, 8)
- `output_path`: Save location (optional)
- `show`: Display plot (default: True)

---

## Combined Visualization Tool

### **visualize_sample.py**
Visualizes all three modalities for a single sample in one command.

**Usage:**
```bash
# Display all three modalities for sample 1
cd Data_processing
python visualize_sample.py 1

# Save to specific data root
python visualize_sample.py 1 --data-root /path/to/data

# Save without displaying (for batch processing)
python visualize_sample.py 1 --no-show
```

**Features:**
- Loads all three sample files (heatmap, impedance, occupancy)
- Displays status for each modality
- Handles missing files gracefully
- Automatic path resolution

---

## Example Workflow

```python
# Individual visualizations
from Data_processing.heatmap import visualize_heatmap
from Data_processing.impedance import visualize_impedance
from Data_processing.occupancy import visualize_occupancy_grid

sample_id = 1
root = Path("src/data_2")

# Visualize heatmap (2 channels)
visualize_heatmap(root / "heatmap" / f"sample_{sample_id}.npy")

# Visualize impedance (log-log plot)
visualize_impedance(root / "Imp" / f"sample_{sample_id}.npy")

# Visualize occupancy grid (7x8)
visualize_occupancy_grid(root / "Occ_map" / f"sample_{sample_id}.npy")
```

---

## Output Formats

All visualization functions support:
- **PNG**: Save with `output_path='path/to/file.png'`
- **Display**: Show in interactive window with `show=True` (default)
- **Batch**: Suppress display with `show=False` for scripting

**DPI Setting:**
- Default: 150 DPI (good for screen viewing)
- Modify in function calls for higher quality

---

## File Statistics

| Module | Lines | Functions | Visualization |
|--------|-------|-----------|----------------|
| heatmap.py | 87 | create_Heatmaps, load_mask_board, **visualize_heatmap** | ✓ |
| impedance.py | 89 | read_impedance_file, **visualize_impedance** | ✓ |
| occupancy.py | 72 | create_occupancy_grid, **visualize_occupancy_grid** | ✓ |
| visualize_sample.py | 62 | visualize_sample (combined) | ✓ |
| Data_processing.py | 130 | Dataset pipeline | - |
| **Total** | **440** | | |
