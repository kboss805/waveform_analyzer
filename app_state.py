"""
Application state manager for the Waveform Analyzer.

This module manages global and per-waveform state without UI or calculation logic.
"""

from typing import List, Optional, Tuple

# Default values
DEFAULT_DURATION = 10.0  # seconds

# Parameter bounds
DURATION_MIN = 0.5
DURATION_MAX = 120.0
DURATION_STEP = 0.5

FREQ_MIN = 0.1
FREQ_MAX = 100.0
FREQ_STEP = 0.1

AMP_MIN = 0.0
AMP_MAX = 10.0
AMP_STEP = 0.1

DUTY_MIN = 1.0
DUTY_MAX = 100.0
DUTY_STEP = 1.0

OFFSET_MIN = 0.0
OFFSET_MAX = 10.0
OFFSET_STEP = 0.1


class WfState:
    """Manages state for a single waveform."""

    def __init__(
        self,
        wf_id: int,
        wf_type: str = "sine",
        freq: float = 0.2,
        amp: float = 2.0,
        offset: float = 8.0,
        duty_cycle: float = 50.0,
        color: Tuple[int, int, int] = (255, 255, 0),
        enabled: bool = True,
        name: str = ""
    ):
        """
        Initialize waveform state.

        Args:
            wf_id: Unique identifier (0-4)
            wf_type: Type of waveform (sine, square, sawtooth, triangle)
            freq: Frequency in Hz (0.1-100.0)
            amp: Amplitude (0.0-10.0)
            offset: Y-axis offset (0.0-10.0)
            duty_cycle: Duty cycle percentage (1.0-100.0, for Square only)
            color: RGB color tuple
            enabled: Whether waveform is visible
            name: Custom display name (empty string uses default)
        """
        self.id = wf_id
        self.wf_type = wf_type
        self.freq = max(FREQ_MIN, min(FREQ_MAX, freq))
        self.amp = max(AMP_MIN, min(AMP_MAX, amp))
        self.offset = max(OFFSET_MIN, min(OFFSET_MAX, offset))
        self.duty_cycle = max(DUTY_MIN, min(DUTY_MAX, duty_cycle))
        self.color = color
        self.enabled = enabled
        self.name = name

    @property
    def display_name(self) -> str:
        """Return custom name if set, otherwise default name."""
        return self.name if self.name else f"Waveform {self.id + 1}"


class AppState:
    """Manages global application state."""

    # Color palette for auto-assignment
    COLORS = [
        (255, 255, 0),    # Yellow
        (0, 255, 255),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 0),      # Green
        (255, 165, 0)     # Orange
    ]

    MAX_WFS = 5
    MIN_WFS = 1

    def __init__(self):
        """Initialize application state with default values."""
        self.duration: float = DEFAULT_DURATION  # 0.5-120.0 seconds
        self.sample_rate: int = 1000  # Fixed
        self.active_wf_index: int = 0
        self.show_max_env: bool = False
        self.show_min_env: bool = False
        self.hide_src_wfs: bool = False

        # Initialize with one default sine waveform
        self.wfs: List[WfState] = [
            WfState(
                wf_id=0,
                wf_type="sine",
                freq=0.2,
                amp=2.0,
                color=self.COLORS[0],
                enabled=True
            )
        ]

    def add_wf(self) -> Optional[WfState]:
        """
        Add a new waveform if under max limit.

        Returns:
            The newly created WfState, or None if at max limit
        """
        if len(self.wfs) >= self.MAX_WFS:
            return None

        wf_id = len(self.wfs)
        color = self.COLORS[wf_id % len(self.COLORS)]

        new_wf = WfState(
            wf_id=wf_id,
            wf_type="sine",
            freq=0.2,
            amp=2.0,
            color=color,
            enabled=True
        )

        self.wfs.append(new_wf)
        self.active_wf_index = wf_id

        return new_wf

    def remove_wf(self, wf_id: int) -> bool:
        """
        Remove a waveform if above min limit.

        Args:
            wf_id: ID of waveform to remove

        Returns:
            True if removed, False if at min limit or not found
        """
        if len(self.wfs) <= self.MIN_WFS:
            return False

        # Find and remove waveform
        self.wfs = [w for w in self.wfs if w.id != wf_id]

        # Reassign IDs and colors
        for idx, wf in enumerate(self.wfs):
            wf.id = idx
            wf.color = self.COLORS[idx % len(self.COLORS)]

        # Update active index if needed
        if self.active_wf_index >= len(self.wfs):
            self.active_wf_index = len(self.wfs) - 1

        return True

    def get_wf(self, wf_id: int) -> Optional[WfState]:
        """
        Get waveform by ID.

        Args:
            wf_id: ID of waveform to retrieve

        Returns:
            WfState or None if not found
        """
        for wf in self.wfs:
            if wf.id == wf_id:
                return wf
        return None

    def get_active_wf(self) -> Optional[WfState]:
        """
        Get currently active waveform.

        Returns:
            Active WfState or None
        """
        if 0 <= self.active_wf_index < len(self.wfs):
            return self.wfs[self.active_wf_index]
        return None

    def get_enabled_wfs(self) -> List[WfState]:
        """
        Get list of enabled waveforms.

        Returns:
            List of enabled WfState objects
        """
        return [w for w in self.wfs if w.enabled]

    def can_show_envelopes(self) -> bool:
        """
        Check if envelopes can be shown (requires >1 enabled waveform).

        Returns:
            True if envelopes can be displayed
        """
        return len(self.get_enabled_wfs()) > 1

    def set_duration(self, duration: float) -> None:
        """
        Set duration with bounds checking.

        Args:
            duration: Duration in seconds (0.5-120.0)
        """
        self.duration = max(DURATION_MIN, min(DURATION_MAX, duration))


# Global singleton instance
app_state = AppState()
