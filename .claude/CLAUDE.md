# Waveform Analyzer

**Version:** 1.3.0
**Target Users:** Engineers visualizing up to 5 independent waveforms with envelope analysis  
**Tech Stack:** Python 3.11+, NumPy 1.24+, SciPy 1.11+, CustomTkinter 5.2+, CTkMenuBar 0.9+, matplotlib 3.8+

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
| `main.py` | Entry point | CustomTkinter context, main loop | No UI creation, no business logic |
| `app_state.py` | State manager | Global state, waveform state | No UI, no calculations |
| `waveform_generator.py` | Pure functions | Wave/envelope generation | No state, no UI |
| `ui_components.py` | UI layer | CustomTkinter widgets, matplotlib plots, callbacks | No calculations |
| `data_export.py` | Export logic | CSV generation | No UI |
| `config.py` | Configuration | Load/save `default.cfg` (INI), fallback defaults | No UI, no state mutation |

### Data Flow (Core Pattern)
```
User Input → Callback → Update State → Regenerate Waveforms → Update UI → Render
                                    ↓
                            Compute Envelopes (if enabled)
```

### Design Constraints (Never Violate)
- ✅ CustomTkinter for the UI (modern Windows 11 look)
- ✅ matplotlib for plotting (dark theme)
- ✅ Native OS file dialogs for export
- ✅ NumPy only (no Pandas for CSV)
- ✅ All waveforms share same time base
- ✅ Separate UI from logic/calculations
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

## Completed Requirements (v1.0)

All v1.0 user stories completed. See git history for original acceptance criteria.

| Story | Feature | Key Criteria |
|-------|---------|-------------|
| US1 | Multiple Waveforms | Real-time display, >30 FPS, <100ms updates, pan/zoom |
| US2 | Dynamic Parameters | Freq 0.1-100 Hz, Amp 0-10, Duty 1-100%, Duration 0.5-120s |
| US3 | Envelope Generation | Max/Min checkboxes, glow lines, real-time, auto-disable when ≤1 wf |
| US4 | CSV Export | Time + amplitude columns, metadata headers, native file dialog |

---

## Current Feature Set (v1.2)

### Supported Waveforms
| Type | Required Parameters | Optional Parameters |
|------|-------------------|-------------------|
| Sine | frequency (0.1-100 Hz), amplitude (0-10) | - |
| Square | frequency, amplitude | duty_cycle (1-100%, default 50%) |
| Sawtooth | frequency, amplitude | - |
| Triangle | frequency, amplitude | - |

**Limits:** 5 waveforms max, 1 waveform minimum

### Envelope Analysis
- **Max Envelope:** Green (#00FF00) glowing line - highest amplitude at each time sample
- **Min Envelope:** Red (#FF0000) glowing line - lowest amplitude at each time sample
- **RMS Envelope:** Orange (#FFA500) glowing line - root-mean-square across all enabled waveforms
- **Peak-to-Peak Fill:** Cyan (#00FFFF) shaded region between max and min envelopes (appears automatically when both are enabled)
- **Glow Effect:** Layered lines (8px/6px/4px at 0.1/0.2/0.3 alpha + 2px core at full alpha)
- **Auto-hide:** Source waveforms are automatically hidden when any envelope is enabled
- **Behavior:** Disabled when ≤1 waveform, calculates from enabled waveforms only, real-time updates

### Measurement Cursors
- **Always On:** Live cursor tracking is always active (no toggle button)
- **Live Tracking:** A cursor line follows the mouse over the plot
- **Proximity Highlight:** When the cursor is near a visible line (waveform or envelope), the cursor color matches that line and a highlight dot appears at the intersection point
- **Pin Reference:** Left-click on the plot to pin a dashed gray reference cursor for comparison
- **Persistence:** Pinned cursor survives parameter changes (re-drawn after plot update)
- **No Sidebar Controls:** All cursor feedback is displayed directly on the plot

### Waveform Renaming & Color Customization
- **Right-click** any waveform button to open context menu with "Rename..." and "Change Color..." options
- **Tooltip** on hover: "Right-click to rename or change color"
- Custom names propagate to plot legend and CSV export
- Duplicate names are rejected with a re-prompt
- Empty name reverts to default ("Waveform N")
- **Change Color...** opens a system color chooser; custom colors survive waveform removal

### Configuration
- **File:** `default.cfg` (INI format) stored alongside the application or executable
- **Access:** File → Configure... opens a modal dialog (420×640)
- **Sections:**
  - **Global** — `duration` (wave duration in seconds)
  - **Waveform** — `type`, `frequency`, `amplitude`, `offset`, `duty_cycle` (applied on next launch)
  - **Display** — `y_axis_title`, `y_min`, `y_max`, `theme` (applied immediately on save)
- **Behavior:** Display settings update the live plot instantly; waveform settings are read at `app_state` import time
- **PyInstaller:** Config path resolves to the directory of the executable when frozen

### Theme Toggle
- **Access:** File → Toggle Theme
- **Modes:** Dark (default) and Light
- **Scope:** Switches CustomTkinter appearance mode, matplotlib plot style, menu bar, and all UI colors
- **Persistence:** Theme choice stored in `default.cfg` under `[display] theme = dark|light`
- **Implementation:** Two theme dicts (`DARK_THEME` / `LIGHT_THEME`) in `ui_components.py`; `_theme` module-level reference swapped on toggle

### Export Capability
- **Formats:** CSV, MATLAB .mat, JSON (selected via file dialog extension filter)
- **CSV:** Time + amplitude columns with metadata header comments
- **MATLAB .mat:** Named variables per waveform/envelope, metadata struct (via `scipy.io.savemat`)
- **JSON:** Structured document with time array, waveform objects (params + data), envelope objects
- **Scope:** Exports all enabled waveforms and active envelopes

---

## State Model

### Per-Waveform State (`WfState` in `app_state.py`)
```python
{
    "id": int,              # 0-4
    "wf_type": str,         # "sine" | "square" | "sawtooth" | "triangle"
    "freq": float,          # 0.1-100.0 Hz
    "amp": float,           # 0.0-10.0
    "offset": float,        # 0.0-10.0
    "duty_cycle": float,    # 1.0-100.0% (Square only)
    "color": tuple,         # (R, G, B) auto-assigned
    "enabled": bool,        # Show/hide waveform
    "name": str             # Custom display name (empty = default)
}
```

### Global State (`AppState` in `app_state.py`)
```python
{
    "duration": float,          # 0.5-120.0 seconds (wave duration)
    "sample_rate": int,         # 1000 (fixed)
    "active_wf_index": int,     # Which waveform controls are shown
    "show_max_env": bool,       # MaxEnvelope visibility
    "show_min_env": bool,       # MinEnvelope visibility
    "show_rms_env": bool,       # RMS Envelope visibility
    "hide_src_wfs": bool        # Auto-set when envelopes enabled
}
```

### Initial State (Startup)
- Values loaded from `default.cfg`; built-in fallbacks used if file is missing
- Default fallback: 1 Sine waveform, 0.2 Hz, 2.0 amplitude, 8.0 offset, Yellow, enabled
- Default fallback: duration 10.0s, Envelopes: OFF

---

## UI Specification

### Layout (1200x900 default, 1000x800 minimum, Dark Theme #202020)

**Menu Bar:** File > Configure... | Toggle Theme, Help > About... (CTkMenuBar, themed)

**Sidebar (330px, scrollable):**
```
┌─ Waveforms ──────────────────┐
│ [+ Add Waveform]             │  ← Max 5
│ [Waveform 1    ] [ON] [X]    │  ← Click=select, ON/OFF=toggle, X=remove
│ [Waveform 2    ] [OFF][X]    │     Right-click=rename, hover=tooltip
├─ Waveform Parameters ────────┤  ← Shows selected waveform
│ Type: [Sine ▼]               │
│ Wave Duration: [input][-][+] │  ← 0.5-120 seconds
│ Offset: [input] [-] [+]      │
│ Frequency: [input] [-] [+]   │
│ Amplitude: [input] [-] [+]   │
│ Duty Cycle: (hidden for Sine)│
├─ Advanced ───────────────────┤
│ ☐ Show Max Envelope          │  ← Disabled when ≤1 waveform
│ ☐ Show Min Envelope          │  ← Peak-to-Peak fill when both on
│ ☐ Show RMS Envelope          │  ← Orange glow line
├─ Export ─────────────────────┤
│ [Export to CSV]              │  ← Opens native OS file dialog
│ Status: Ready                │
└──────────────────────────────┘
```

**Main Area:** Single plot, all waveforms overlaid, shared time axis

**Status Bar:** `Waveforms: X/5`

**About Dialog:** App name, version (APP_VERSION constant), description, author info

### Color Palette (Auto-assigned, User-customizable via Right-Click)
| Waveform | Color | RGB |
|----------|-------|-----|
| 1 | Light Blue 300 | (79, 195, 247) |
| 2 | Pink 300 | (240, 98, 146) |
| 3 | Deep Orange 300 | (255, 138, 101) |
| 4 | Green 400 | (102, 187, 106) |
| 5 | Deep Purple 200 | (179, 157, 219) |
| MaxEnvelope | Green (glow) | #6CCB5F / #0E7A0D |
| MinEnvelope | Red (glow) | #FF99A4 / #C42B1C |
| RMSEnvelope | Amber (glow) | #FFB900 / #D88300 |
| Peak-to-Peak | Teal (fill) | #80CBC4 / #009688 @ 12% alpha |
| Live Cursor | White (default) | #FFFFFF @ 50% alpha, matches nearest line on proximity |
| Pinned Cursor | Gray (dashed) | #BDBDBD @ 70% alpha |

### Plot Styling
- Background: #1C1C1C (dark) / #FFFFFF (light), Grid: themed @ 30% opacity
- Line width: 2px, Anti-aliasing: enabled

---

## Implementation Reference

### Wave Generation Functions (`waveform_generator.py`)
```python
import numpy as np
from scipy import signal

# All functions return Tuple[np.ndarray, np.ndarray] (time, amplitude)
# Waveforms use half-amplitude centered on offset: offset + (amp/2) * waveform()

def gen_sine_wf(freq, amp, offset=0.0, dur=1.0, sample_rate=1000):
    time = np.linspace(0, dur, int(sample_rate * dur))
    return time, offset + (amp / 2) * np.sin(2 * np.pi * freq * time)

def gen_square_wf(freq, amp, duty_cycle, offset=0.0, dur=1.0, sample_rate=1000):
    time = np.linspace(0, dur, int(sample_rate * dur))
    return time, offset + (amp / 2) * signal.square(2 * np.pi * freq * time, duty=duty_cycle/100)

def gen_sawtooth_wf(freq, amp, offset=0.0, dur=1.0, sample_rate=1000):
    time = np.linspace(0, dur, int(sample_rate * dur))
    return time, offset + (amp / 2) * signal.sawtooth(2 * np.pi * freq * time)

def gen_triangle_wf(freq, amp, offset=0.0, dur=1.0, sample_rate=1000):
    time = np.linspace(0, dur, int(sample_rate * dur))
    return time, offset + (amp / 2) * signal.sawtooth(2 * np.pi * freq * time, width=0.5)

# Dispatcher — primary API used by ui_components.py
def gen_wf(wf_type, freq, amp, offset=0.0, duty_cycle=50.0, dur=1.0, sample_rate=1000):
    # Routes to gen_sine_wf / gen_square_wf / gen_sawtooth_wf / gen_triangle_wf
    # Defaults to sine if wf_type is unrecognized

# Envelope calculations
def compute_max_env(wfs):
    time = wfs[0][0]
    return time, np.max(np.array([w[1] for w in wfs]), axis=0)

def compute_min_env(wfs):
    time = wfs[0][0]
    return time, np.min(np.array([w[1] for w in wfs]), axis=0)

def compute_rms_env(wfs):
    time = wfs[0][0]
    amps = np.array([w[1] for w in wfs])
    return time, np.sqrt(np.mean(amps ** 2, axis=0))
```

### UI Architecture Patterns

**matplotlib Integration:** Themed `Figure` embedded via `FigureCanvasTkAgg` in a CustomTkinter frame. Plot update cycle: `ax.clear()` → plot all waveforms → draw envelopes → `canvas.draw()`.

**Glow Effect:** Layered lines with decreasing linewidth (8/6/4px at 0.1/0.2/0.3 alpha) plus 2px core at full alpha.

**Parameter Controls:** `CTkEntry` for numeric input with paired `CTkButton` (+/-) for increment/decrement. Step values defined as constants in `app_state.py`.

**File Dialogs:** `tkinter.filedialog.asksaveasfilename()` for native OS export dialogs.

**Menu Bar:** `CTkMenuBar` for themed File/Help menus (not standard tkinter Menu).

**Theme System:** WinUI-aligned color tokens defined in `winui_theme.json` (CustomTkinter default theme) and `DARK_THEME`/`LIGHT_THEME` dicts in `ui_components.py`. Font: Segoe UI Variable. Corner radii: 4px controls, 8px containers.

See `ui_components.py` for full implementation details.

### CSV Export Format
```csv
# Exported: 2025-01-03 14:30:00
# MySignal: Sine, 0.2 Hz, 2.0 amp, 8.0 offset
# Waveform_2: Square, 0.2 Hz, 2.0 amp, 8.0 offset, 50.0% duty cycle
# Max_Envelope: Computed from 2 waveforms
# Sample Rate: 1000 S/s, Duration: 10.0s
Time (s),MySignal,Waveform_2,Max_Envelope
0.000000,8.000000,8.000000,8.000000
0.001000,8.001257,9.000000,9.000000
```
Note: Custom waveform names (via rename) are used for column headers. Spaces replaced with underscores.

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

### Automated Tests (106 tests)
Run: `python -m pytest test_waveform_analyzer.py -v`

Covers the pre-commit checklist automatically:
| Test Class | Checklist Item | Tests |
|-----------|----------------|-------|
| `TestWaveTypes` | All 4 wave types render correctly | 19 |
| `TestEdgeCases` | Min/max frequency (0.1/100 Hz), amplitude (0/10) | 7 |
| `TestDutyCycle` | Duty cycle 1%, 50%, 100% for Square | 5 |
| `TestWaveDuration` | Wave duration 0.5s, 10s, 120s | 5 |
| `TestEnvelopes` | Envelopes with 2, 3, 5 waveforms, same/opposite phase | 12 |
| `TestMixedEnabledDisabled` | Mixed enabled/disabled waveforms | 4 |
| `TestEnvelopeToggles` | Toggle max/min independently and together | 6 |
| `TestCSVExport` | CSV export with/without envelopes | 6 |
| `TestWaveformLimits` | Add/remove waveforms (min/max limits) | 8 |
| `TestNoErrors` | No console errors or warnings | 5 |
| `TestPerformance` | FPS >30 (gen <100ms, envelope <10ms) | 3 |
| `TestConfig` | Configuration load/save | 3 |
| `TestMATExport` | MATLAB .mat export with/without envelopes | 5 |
| `TestJSONExport` | JSON export with/without envelopes | 6 |
| `TestColorCustomization` | Color assignment, custom colors, preserve on remove | 4 |
| `TestThemeToggle` | Theme dicts, key consistency, config default | 4 |

### Pre-Commit Checklist (Manual Items)
After running automated tests, verify these UI-dependent items manually:

### Manual Test Scenarios
1. **Startup:** Should show waveform with values from `default.cfg` (or fallback defaults)
2. **Add to max:** Add 4 more waveforms, verify limit enforcement
3. **Remove to min:** Remove to 1 waveform, verify limit enforcement
4. **Envelope edge:** Enable max/min with only 1 wave (should be disabled)
5. **Performance:** Change frequency rapidly, verify smooth updates
6. **Rename:** Right-click waveform, enter custom name, verify in list/legend/CSV
7. **Duplicate name:** Try renaming to an existing name, verify rejection
8. **Auto-hide:** Enable envelope, verify source waveforms hidden automatically
9. **Configure dialog:** Open File → Configure..., change display settings, verify plot updates immediately
10. **Config persistence:** Modify waveform settings in Configure, restart app, verify new values load
11. **Theme toggle:** File → Toggle Theme switches dark/light, verify plot/menu/sidebar update, restart to verify persistence

---

## Future Feature Roadmap

### v1.1 - Enhanced Analysis
- ✅ **Peak-to-Peak Envelope:** Cyan shaded area between max/min (auto when both enabled)
- ✅ **RMS Envelope:** Orange glowing line, RMS across enabled waveforms
- ✅ **Measurement Cursors:** Always-on live tracking cursor with proximity highlight and pinned reference

### v1.2 - Export & Usability
- ✅ **Export Formats:** CSV, MATLAB .mat, JSON (file dialog offers all three)
- ✅ **Color Customization:** Right-click → Change Color... (system color chooser, persists across removes)
- ✅ **Theme Toggle:** File → Toggle Theme switches dark/light mode (persists in `default.cfg`)

### v2.0 - Advanced Features
- [ ] **Statistics Panel:** Mean, RMS, peak-to-peak per waveform
- [ ] **Multiple Plots:** Separate plots option
- [ ] **Detachable Windows:** Resizable, multi-monitor support
- ✅ **Custom User Settings:** Configurable defaults via `default.cfg` and File → Configure... dialog
- ✅ **Sphinx Documentation:** Auto-generated API docs from Google-style docstrings (`docs/`)

---

## Development Workflow

### Before Starting Any Task

Ask yourself:
1. What feature/story does this relate to? (Check Roadmap)
2. What modules will I modify? (See Module Responsibilities)
3. Does this change the state model? (Update State Model section if yes)
4. Will this affect performance SLAs? (Re-test if yes)
5. Does this violate any design constraints? (Discuss alternatives if yes)
6. What edge cases should I test? (Add to Testing Protocol)

If uncertain, **ask for clarification before coding**.

### Adding a New Feature

1. **Update this document** — add to Roadmap, update State Model / UI Spec if needed
2. **Plan architecture** — check if new module needed, data flow changes, performance impact
3. **Implement** — follow Module Responsibilities, Code Style, Error Handling standards
4. **Validate** — run Pre-Commit Checklist, verify performance SLAs, update `APP_VERSION`

### Quick Reference

**Don't:**
- ❌ Use Pandas for CSV (NumPy only)
- ❌ Create separate time bases per waveform
- ❌ Put business logic in UI layer
- ❌ Skip type hints or docstrings
- ❌ Add config keys without updating the Configure dialog and `save_config()`

**Do:**
- ✅ Keep modules focused on single responsibility
- ✅ Clamp invalid inputs instead of crashing
- ✅ Monitor FPS during development
- ✅ Test edge cases and error conditions
