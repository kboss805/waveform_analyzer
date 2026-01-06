# Real-Time Waveform Visualizer

**Version:** 1.0  
**Target Users:** Engineers visualizing up to 5 independent waveforms with envelope analysis  
**Tech Stack:** Python 3.11+, DearPyGui 1.10+, NumPy 1.24+, SciPy 1.11+

---

## Quick Start
```bash
python -m venv venv && venv\Scripts\activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## Architecture (Stable - Don't Change Without Review)

### Module Responsibilities
| Module | Purpose | What It Does | What It Doesn't Do |
|--------|---------|--------------|-------------------|
| `main.py` | Entry point | DearPyGui context, main loop | No UI creation, no business logic |
| `app_state.py` | State manager | Global state, waveform state | No UI, no calculations |
| `waveform_generator.py` | Pure functions | Wave/envelope generation | No state, no UI |
| `ui_components.py` | UI layer | All `dpg.add_*` calls, callbacks | No calculations |
| `data_export.py` | Export logic | CSV generation | No UI |

### Data Flow (Core Pattern)
```
User Input → Callback → Update State → Regenerate Waveforms → Update UI → Render
                                    ↓
                            Compute Envelopes (if enabled)
```

### Design Constraints (Never Violate)
- ✅ DearPyGui only (no matplotlib mixing)
- ✅ NumPy only (no Pandas for CSV)
- ✅ All waveforms share same time base
- ✅ Separate UI from logic/calculations
- ✅ Use tags for all updatable UI elements
- ✅ Windows compatible

### Performance SLAs
| Metric | Target | Critical? |
|--------|--------|-----------|
| Startup time | <2s | No |
| Plot update latency | <100ms | Yes |
| Frame rate | >30 FPS | Yes |
| Envelope calculation | <10ms | Yes |
| Memory usage | <200MB | No |

---

## User Stories (v1.0 Requirements)

### US1: Visualize Multiple Waveforms
**As an** engineer learning signal processing  
**I want to** see multiple waveforms in time domain plots simultaneously  
**So that** I can understand how waveforms behave and interact

**Acceptance Criteria:**
- ✅ Display all plots updating in real-time
- ✅ Smooth updates (>30 FPS)
- ✅ Independent zoom/pan for each plot
- ✅ All plots update within 100ms of parameter change

### US2: Adjust Parameters Dynamically
**As an** engineer  
**I want to** adjust frequency, amplitude, and duty cycle of each waveform  
**So that** I can experiment and see immediate visual feedback

**Acceptance Criteria:**
- ✅ Frequency: 1-100 Hz, slider step 0.1 Hz
- ✅ Amplitude: 0-10, slider step 0.1
- ✅ Duty Cycle: 1-100%, slider step 1% (only for Square waves)
- ✅ Time Span: 0.1-120 seconds, slider step 0.5s
- ✅ All plots update immediately on parameter change (<100ms latency)
- ✅ Current values displayed clearly next to each slider

### US3: Generate Envelope Waveforms
**As an** engineer  
**I want to** optionally generate envelope waveforms from currently displayed waveforms  
**So that** I can view the maximum or minimum amplitude of all waveforms at any given sample

**Acceptance Criteria:**
- ✅ Checkbox control "Show Max Envelope" in Global Controls
- ✅ Checkbox control "Show Min Envelope" in Global Controls
- ✅ MaxEnvelope displays as red dashed line when enabled
- ✅ MinEnvelope displays as blue dashed line when enabled
- ✅ Both can be enabled simultaneously
- ✅ Envelope waveforms update in real-time (<100ms) when:
  - Any waveform parameter changes
  - Waveforms are added/removed
  - Waveform visibility is toggled
- ✅ Envelope waveforms only calculate from enabled/visible waveforms
- ✅ Envelope checkboxes disabled when only 1 waveform exists
- ✅ Envelope waveforms appear in plot legend
- ✅ Envelope waveforms included in CSV export when enabled
- ✅ No performance degradation (maintain >30 FPS)

### US4: Export Data
**As a** researcher  
**I want to** export waveform data to CSV  
**So that** I can analyze it in other tools or share with colleagues

**Acceptance Criteria:**
- ✅ Export includes time and amplitude columns
- ✅ Metadata in CSV header (waveform type, frequency, amplitude)
- ✅ User can specify filename
- ✅ Success confirmation displayed
- ✅ CSV format compatible with Excel/MATLAB

---

## Current Feature Set (v1.0)

### Supported Waveforms
| Type | Required Parameters | Optional Parameters |
|------|-------------------|-------------------|
| Sine | frequency (1-100 Hz), amplitude (0-10) | - |
| Square | frequency, amplitude | duty_cycle (1-100%, default 50%) |
| Sawtooth | frequency, amplitude | - |
| Triangle | frequency, amplitude | - |

**Limits:** 5 waveforms max, 1 waveform minimum

### Envelope Analysis
- **Max Envelope:** Red (#FF0000) dashed line - highest amplitude at each time sample
- **Min Envelope:** Blue (#0000FF) dashed line - lowest amplitude at each time sample
- **Behavior:** Disabled when ≤1 waveform, calculates from enabled waveforms only, real-time updates

### Export Capability
- **Format:** CSV with time column + amplitude columns
- **Metadata:** Timestamp, waveform parameters, sample rate, duration
- **Options:** Select which waveforms to export (including envelopes)

---

## State Model

### Per-Waveform State
```python
{
    "id": int,              # 0-4
    "wave_type": str,       # "sine" | "square" | "sawtooth" | "triangle"
    "frequency": float,     # 1.0-100.0 Hz
    "amplitude": float,     # 0.0-10.0
    "duty_cycle": float,    # 1.0-100.0% (Square only)
    "color": tuple,         # (R, G, B) auto-assigned
    "enabled": bool         # Show/hide waveform
}
```

### Global State
```python
{
    "time_span": float,              # 1-100.0 seconds
    "sample_rate": int,              # 1000 (fixed)
    "active_waveform_index": int,    # Which waveform controls are shown
    "show_max_envelope": bool,       # MaxEnvelope visibility
    "show_min_envelope": bool        # MinEnvelope visibility
}
```

### Initial State (Startup)
- 1 Sine waveform: 5.0 Hz, 5.0 amplitude, Yellow, enabled
- time_span: 1.0s, Grid: ON, Auto-scale: ON, Envelopes: OFF

---

## UI Specification

### Layout (1200x800 minimum, Dark Theme #1a1a1a)

**Sidebar (300px):**
```
┌─ Global Controls ────────────┐
│ Time Span: [====|=====] 1.0s │
│ ☑ Auto-scale Y-axis          │
│ ☑ Show Grid                  │
│ ☐ Show MaxEnvelope (disabled)│  ← Disabled when ≤1 waveform
│ ☐ Show MinEnvelope (disabled)│
├─ Waveforms ──────────────────┤
│ [+ Add Waveform]             │  ← Max 5
│ ● Sine 5.0 Hz        👁 [X]  │  ← Click=select, eye=toggle, X=remove
│ ○ Square 10.0 Hz     👁 [X]  │  ← Grayed if disabled
├─ Waveform Controls ──────────┤  ← Shows selected waveform
│ Type: [Sine ▼]               │
│ Frequency: [====|===] 5.0 Hz │
│ Amplitude: [====|===] 5.0    │
│ Duty Cycle: (hidden for Sine)│
├─ Export ─────────────────────┤
│ Filename: [waveforms.csv]    │
│ ☑ Waveform 1 ☑ Waveform 2   │
│ ☑ Max Envelope (if enabled)  │
│ [Export]                     │
│ Status: Ready                │
└──────────────────────────────┘
```

**Main Area:** Single plot, all waveforms overlaid, shared time axis

**Status Bar:** `Sample Rate: 1000 S/s | FPS: XX | Waveforms: X/5`

### Color Palette (Auto-assigned)
| Waveform | Color | RGB |
|----------|-------|-----|
| 1 | Yellow | (255, 255, 0) |
| 2 | Cyan | (0, 255, 255) |
| 3 | Magenta | (255, 0, 255) |
| 4 | Green | (0, 255, 0) |
| 5 | Orange | (255, 165, 0) |
| MaxEnvelope | Red (dashed) | (255, 0, 0) |
| MinEnvelope | Blue (dashed) | (0, 0, 255) |

### Plot Styling
- Background: #1a1a1a, Axes: #666666, Grid: #666666 @ 20% opacity
- Line width: 2px, Anti-aliasing: enabled

---

## Implementation Reference

### Wave Generation Functions
```python
import numpy as np
from scipy import signal

# All functions return (time: np.ndarray, amplitude: np.ndarray)

def generate_sine_wave(frequency: float, amplitude: float, duration: float = 1.0, sample_rate: int = 1000):
    time = np.linspace(0, duration, int(sample_rate * duration))
    return time, amplitude * np.sin(2 * np.pi * frequency * time)

def generate_square_wave(frequency: float, amplitude: float, duty_cycle: float, duration: float = 1.0, sample_rate: int = 1000):
    time = np.linspace(0, duration, int(sample_rate * duration))
    return time, amplitude * signal.square(2 * np.pi * frequency * time, duty=duty_cycle/100)

def generate_sawtooth_wave(frequency: float, amplitude: float, duration: float = 1.0, sample_rate: int = 1000):
    time = np.linspace(0, duration, int(sample_rate * duration))
    return time, amplitude * signal.sawtooth(2 * np.pi * frequency * time)

def generate_triangle_wave(frequency: float, amplitude: float, duration: float = 1.0, sample_rate: int = 1000):
    time = np.linspace(0, duration, int(sample_rate * duration))
    return time, amplitude * signal.sawtooth(2 * np.pi * frequency * time, width=0.5)

# Envelope calculations
def compute_max_envelope(waveforms: list[tuple[np.ndarray, np.ndarray]]):
    if not waveforms: return np.array([]), np.array([])
    time = waveforms[0][0]  # Shared time base
    return time, np.max(np.array([w[1] for w in waveforms]), axis=0)

def compute_min_envelope(waveforms: list[tuple[np.ndarray, np.ndarray]]):
    if not waveforms: return np.array([]), np.array([])
    time = waveforms[0][0]  # Shared time base
    return time, np.min(np.array([w[1] for w in waveforms]), axis=0)
```

### DearPyGui Patterns
```python
# Create plot with series
with dpg.plot(label="Waveforms", height=-1, width=-1):
    dpg.add_plot_legend()
    dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)")
    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag="y_axis")
    dpg.add_line_series([], [], label="Waveform 1", parent="y_axis", tag="wave1_series")

# Update existing series
dpg.set_value("wave1_series", [time.tolist(), amplitude.tolist()])

# Slider with callback
dpg.add_slider_float(
    label="Frequency (Hz)", 
    default_value=5.0, 
    min_value=1.0, 
    max_value=100.0,
    callback=on_frequency_changed,
    tag="freq_slider"
)
```

### CSV Export Format
```csv
# Exported: 2025-01-03 14:30:00
# Waveform_1: Sine, 5.0 Hz, 5.0 amplitude
# Waveform_2: Square, 10.0 Hz, 8.0 amplitude, 50% duty cycle
# Max_Envelope: Computed from 2 waveforms
# Sample Rate: 1000 S/s, Duration: 1.0s
Time (s),Waveform_1_Sine,Waveform_2_Square,Max_Envelope
0.000000,0.000000,8.000000,8.000000
0.001000,0.031411,8.000000,8.000000
```

---

## Development Standards

### Code Style
- **PEP 8** compliance, max 88 chars/line
- **Type hints** on all function signatures
- **Google-style docstrings** for public functions
- Functions <30 lines (split if longer)
- Descriptive variable names (no `a`, `b`, `x` unless math context)

### Error Handling
| Error Type | Strategy |
|------------|----------|
| Invalid parameter range | Silently clamp to valid range |
| File export failure | Show error in UI status area |
| Invalid filename | Auto-correct (remove invalid chars, add .csv) |
| Division by zero | Handle gracefully in calculations |
| No enabled waveforms | Hide envelope plots |

---

## Testing Protocol

### Pre-Commit Checklist
- [ ] All 5 wave types render correctly
- [ ] Edge cases: min/max frequency (1 Hz, 100 Hz), min/max amplitude (0, 10)
- [ ] Duty cycle: 1%, 50%, 100% for Square
- [ ] Time span: 0.1s, 1s, 10s
- [ ] Envelopes: Test with 2, 3, 5 waveforms (same phase, opposite phase)
- [ ] Mixed enabled/disabled waveforms
- [ ] Toggle max/min independently and together
- [ ] CSV export with/without envelopes
- [ ] Add/remove waveforms (test min/max limits)
- [ ] No console errors or warnings
- [ ] FPS >30 during parameter changes

### Manual Test Scenarios
1. **Startup:** Should show 1 Sine wave, 1 Hz, 5.0 amplitude
2. **Add to max:** Add 4 more waveforms, verify limit enforcement
3. **Remove to min:** Remove to 1 waveform, verify limit enforcement
4. **Envelope edge:** Enable max/min with only 1 wave (should be disabled)
5. **Performance:** Change frequency rapidly, verify smooth updates

---

## Future Feature Roadmap

### v1.1 - Enhanced Analysis
- [ ] **Peak-to-Peak Envelope:** Shaded area between max/min
- [ ] **RMS Envelope:** Root-mean-square calculation
- [ ] **Moving Average Envelope:** Smoothing filter
- [ ] **Measurement Cursors:** Click to see exact X/Y values

### v1.2 - Composite Operations
- [ ] **Arithmetic Operations:** Add, subtract, multiply, divide waveforms
- [ ] **FFT Display:** Frequency domain analysis
- [ ] **Phase Relationships:** Show phase difference between waveforms

### v1.3 - Export & Usability
- [ ] **Export Formats:** MATLAB .mat, JSON
- [ ] **Auto-save:** Periodic state saving
- [ ] **Keyboard Shortcuts:** Space, A, 1-5 for quick actions
- [ ] **Color Customization:** User-selectable waveform colors
- [ ] **Theme Toggle:** Dark/light mode

### v2.0 - Advanced Features
- [ ] **Statistics Panel:** Mean, RMS, peak-to-peak per waveform
- [ ] **Trigger Mode:** Oscilloscope-style triggering
- [ ] **Multiple Plots:** Separate plots option
- [ ] **Detachable Windows:** Resizable, multi-monitor support
- [ ] **Custom User Settings:** Allow user to define default waveform parameters, display values, etc. from a configuration file 

---

## Adding New Features (Guide for Future Development)

### Step 1: Update This Document
1. Add feature to appropriate version section in Roadmap
2. Update State Model if new state required
3. Update UI Specification if new controls needed
4. Document any new constraints

### Step 2: Plan Architecture Impact
- Does it need new module? Update Project Structure
- Does it change data flow? Update Architecture section
- Does it affect performance SLAs? Update metrics
- Does it violate design constraints? Discuss alternatives

### Step 3: Implement
- Create functions in appropriate modules per Module Responsibilities
- Follow Code Style and Error Handling standards
- Update Testing Protocol with new test cases

### Step 4: Validate
- Run full Testing Protocol checklist
- Verify performance SLAs still met
- Update version number if releasing

---

## Common Pitfall Avoidance

### Don't:
- ❌ Mix matplotlib with DearPyGui
- ❌ Use Pandas for CSV (NumPy only)
- ❌ Create separate time bases per waveform
- ❌ Put business logic in UI layer
- ❌ Use global variables excessively
- ❌ Skip type hints or docstrings
- ❌ Create config files without requirement

### Do:
- ✅ Keep modules focused on single responsibility
- ✅ Separate UI creation from callbacks/logic
- ✅ Use tags for all updatable UI elements
- ✅ Test edge cases and error conditions
- ✅ Monitor FPS during development
- ✅ Clamp invalid inputs instead of crashing
- ✅ Write descriptive commit messages

---

## Questions for Claude When Starting Work

Before implementing ANY task, ask:
1. **What user story does this relate to?**
2. **What modules will I modify?**
3. **Does this change the state model?**
4. **Will this affect performance SLAs?**
5. **What edge cases should I test?**
6. **Does this violate any design constraints?**

If uncertain, **ask for clarification before coding**.
