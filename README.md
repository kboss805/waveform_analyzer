# Waveform Analyzer

A Python application for visualizing and analyzing multiple waveforms in real-time with envelope analysis capabilities.

## Installation

1. **Activate the virtual environment:**
   ```bash
   # Windows
   env\Scripts\activate
   ```

2. **Install dependencies (if not already done):**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
python main.py
```

Or with the virtual environment:
```bash
env\Scripts\python.exe main.py
```

## Quick Start Guide

### Basic Usage

1. **Launch the application** - The app starts with one default sine wave (0.2 Hz, amplitude 2.0, offset 8.0)

2. **Adjust waveform parameters:**
   - Use the **Frequency** input to change the wave frequency (0.1-100 Hz)
   - Use the **Amplitude** input to adjust the wave amplitude (0-10)
   - Use the **Offset** input to adjust the Y-axis offset (0-10)
   - For Square waves, use the **Duty Cycle** input (1-100%)

3. **Change waveform type:**
   - Use the **Type** dropdown to select: Sine, Square, Sawtooth, or Triangle

4. **Add more waveforms:**
   - Click **+ Add Waveform** button (max 5 waveforms)
   - Each waveform gets a unique color automatically

5. **Manage waveforms:**
   - Click on a waveform in the list to select and edit it
   - Click the **ON/OFF button** to show/hide a waveform
   - Click the **X button** to remove a waveform (minimum 1 required)
   - **Right-click** a waveform button to rename it (custom names appear in the plot legend and CSV export)

### Envelope Analysis

When you have 2 or more **enabled** waveforms:

- **Show Max Envelope** - Displays the maximum amplitude at each time point (green glowing line)
- **Show Min Envelope** - Displays the minimum amplitude at each time point (red glowing line)

Both envelopes can be enabled simultaneously. Source waveforms are automatically hidden when envelopes are shown.

### Waveform Parameters

Each waveform has the following adjustable parameters:

- **Type** - Waveform type (Sine, Square, Sawtooth, Triangle)
- **Wave Duration** - Adjusts the time window for visualization (0.5-120 seconds)
- **Frequency** - Wave frequency in Hz (0.1-100 Hz)
- **Amplitude** - Wave amplitude (0-10)
- **Offset** - Y-axis offset (0-10)
- **Duty Cycle** - For Square waves only (1-100%)

### Exporting Data

1. Click **Export to CSV**
2. A native file dialog will open - choose your save location and filename
3. Exported data includes:
   - All enabled waveforms
   - Active envelopes (if enabled)
   - Metadata (waveform type, frequency, amplitude, etc.)

## Features

- **4 Waveform Types:** Sine, Square, Sawtooth, Triangle
- **Up to 5 simultaneous waveforms**
- **Custom waveform naming** via right-click context menu
- **Real-time visualization** with responsive performance
- **Envelope analysis** with glowing Max/Min envelope lines
- **Auto-hide source waveforms** when envelopes are enabled
- **CSV export** with native OS file dialog (includes custom names)
- **Independent waveform control**
- **Help menu** with About dialog (version, author info)
- **Modern Windows 11 UI** with CustomTkinter
- **Dark theme** matplotlib plotting

## File Structure

```
waveform_generator/
├── main.py                  # Application entry point
├── app_state.py            # State management
├── waveform_generator.py   # Waveform generation functions
├── ui_components.py        # UI and callbacks
├── data_export.py          # CSV export functionality
├── requirements.txt        # Python dependencies
├── CLAUDE.md              # Full specification document
└── README.md              # This file
```

## System Requirements

- Python 3.11 or higher
- Windows/Linux/macOS
- Minimum resolution: 1000x800 (recommended: 1200x950)

## Dependencies

- **CustomTkinter** >= 5.2.0 - Modern GUI framework
- **CTkMenuBar** >= 0.9 - Dark-themed menu bar for CustomTkinter
- **matplotlib** >= 3.8.0 - Plotting library
- **NumPy** >= 1.24.0 - Numerical computing
- **SciPy** >= 1.11.0 - Signal processing

## Performance

- **Target frame rate:** >30 FPS
- **Plot update latency:** <100ms
- **Memory usage:** <200MB
- **Startup time:** <2 seconds

## Keyboard Shortcuts

- Use +/- buttons for fine control of parameters
- Click and drag on the plot to pan
- Scroll to zoom in/out on the plot

## Troubleshooting

**Application won't start:**
- Ensure Python 3.11+ is installed: `python --version`
- Verify dependencies are installed: `pip list`
- Activate virtual environment first

**Performance issues:**
- Reduce the number of visible waveforms
- Decrease the wave duration
- Disable envelopes if not needed

**Export fails:**
- Check file path permissions
- Ensure filename doesn't contain invalid characters
- The app will auto-correct the filename and add `.csv` if needed

## Version

Current version: **0.9.1**

See [CLAUDE.md](CLAUDE.md) for the complete specification and future roadmap.
