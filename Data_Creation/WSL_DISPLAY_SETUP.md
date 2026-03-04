# WSL Display Setup for Python Matplotlib

## Why plots don't display in WSL

WSL (Windows Subsystem for Linux) doesn't have a native display server. Python's matplotlib needs an X11 server to display GUI windows.

## Solution: Install and Configure X Server

### Option 1: VcXsrv (Recommended)

1. **Download and Install VcXsrv** (on Windows):
   - Download from: https://sourceforge.net/projects/vcxsrv/
   - Install with default settings

2. **Launch XLaunch**:
   - Search for "XLaunch" in Windows Start menu
   - Select "Multiple windows"
   - Display number: 0
   - Start no client
   - **IMPORTANT**: Check "Disable access control"
   - Finish

3. **Set DISPLAY variable in WSL**:
   ```bash
   # Add to ~/.bashrc
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
   
   # Or for WSL2, use:
   export DISPLAY=$(ip route list default | awk '{print $3}'):0.0
   
   # Apply changes
   source ~/.bashrc
   ```

4. **Test the setup**:
   ```bash
   # Install x11-apps if not installed
   sudo apt-get install x11-apps
   
   # Test with simple X app
   xclock
   ```

### Option 2: Windows 11 WSLg (Built-in)

If you're on Windows 11 with WSLg (GUI support built-in):

```bash
# WSLg should work automatically, just ensure you're updated
wsl --update
```

### Option 3: Use Windows Terminal with Display

```bash
# Add to ~/.bashrc
export DISPLAY=:0
```

## Verify Matplotlib Backend

After setting up X server, verify matplotlib can display:

```python
import matplotlib
print(matplotlib.get_backend())  # Should show 'TkAgg' or 'Qt5Agg'

import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.show()  # Window should appear
```

## Troubleshooting

### Error: "Can't connect to display"
- Make sure VcXsrv is running on Windows
- Check DISPLAY variable: `echo $DISPLAY`
- Verify firewall isn't blocking VcXsrv

### Error: "no display name and no $DISPLAY environment variable"
- Set DISPLAY variable as shown above
- Restart your WSL terminal

### Still not working?
- Try different backends in Python:
  ```python
  import matplotlib
  matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg'
  ```

- Install required packages:
  ```bash
  sudo apt-get install python3-tk
  ```

## Alternative: Just Save Figures

If X server setup is too complex, you can just save plots:

```python
# Modify the viewer to only save instead of show
viewer = InteractiveSampleViewer(data_root="datasets", output_pth="temp_visuals")
```

Or use non-GUI backend:
```python
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
```
