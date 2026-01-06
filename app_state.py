"""
Application state manager for the Waveform Generator/Analyzer.

This module manages global and per-waveform state without UI or calculation logic.
"""

from typing import Dict, List, Optional, Tuple

# Default values
DEFAULT_TIME_SPAN = 10.0  # seconds


class WaveformState:
    """Manages state for a single waveform."""

    def __init__(
        self,
        waveform_id: int,
        wave_type: str = "sine",
        frequency: float = 1.0,
        amplitude: float = 5.0,
        duty_cycle: float = 50.0,
        color: Tuple[int, int, int] = (255, 255, 0),
        enabled: bool = True
    ):
        """
        Initialize waveform state.

        Args:
            waveform_id: Unique identifier (0-4)
            wave_type: Type of waveform (sine, square, pulse, sawtooth, triangle)
            frequency: Frequency in Hz (1.0-100.0)
            amplitude: Amplitude (0.0-10.0)
            duty_cycle: Duty cycle percentage (1.0-100.0, for Square/Pulse only)
            color: RGB color tuple
            enabled: Whether waveform is visible
        """
        self.id = waveform_id
        self.wave_type = wave_type
        self.frequency = max(1.0, min(100.0, frequency))
        self.amplitude = max(0.0, min(10.0, amplitude))
        self.duty_cycle = max(1.0, min(100.0, duty_cycle))
        self.color = color
        self.enabled = enabled

    def to_dict(self) -> Dict:
        """Convert waveform state to dictionary."""
        return {
            "id": self.id,
            "wave_type": self.wave_type,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "duty_cycle": self.duty_cycle,
            "color": self.color,
            "enabled": self.enabled
        }


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

    MAX_WAVEFORMS = 5
    MIN_WAVEFORMS = 1

    def __init__(self):
        """Initialize application state with default values."""
        self.time_span: float = DEFAULT_TIME_SPAN  # 0.1-100.0 seconds
        self.sample_rate: int = 1000  # Fixed
        self.active_waveform_index: int = 0
        self.show_max_envelope: bool = False
        self.show_min_envelope: bool = False
        self.show_grid: bool = True
        self.hide_source_waveforms: bool = False

        # Initialize with one default sine waveform
        self.waveforms: List[WaveformState] = [
            WaveformState(
                waveform_id=0,
                wave_type="sine",
                frequency=1.0,
                amplitude=5.0,
                color=self.COLORS[0],
                enabled=True
            )
        ]

    def add_waveform(self) -> Optional[WaveformState]:
        """
        Add a new waveform if under max limit.

        Returns:
            The newly created WaveformState, or None if at max limit
        """
        if len(self.waveforms) >= self.MAX_WAVEFORMS:
            return None

        waveform_id = len(self.waveforms)
        color = self.COLORS[waveform_id % len(self.COLORS)]

        new_waveform = WaveformState(
            waveform_id=waveform_id,
            wave_type="sine",
            frequency=1.0,
            amplitude=5.0,
            color=color,
            enabled=True
        )

        self.waveforms.append(new_waveform)
        self.active_waveform_index = waveform_id

        return new_waveform

    def remove_waveform(self, waveform_id: int) -> bool:
        """
        Remove a waveform if above min limit.

        Args:
            waveform_id: ID of waveform to remove

        Returns:
            True if removed, False if at min limit or not found
        """
        if len(self.waveforms) <= self.MIN_WAVEFORMS:
            return False

        # Find and remove waveform
        self.waveforms = [w for w in self.waveforms if w.id != waveform_id]

        # Reassign IDs and colors
        for idx, waveform in enumerate(self.waveforms):
            waveform.id = idx
            waveform.color = self.COLORS[idx % len(self.COLORS)]

        # Update active index if needed
        if self.active_waveform_index >= len(self.waveforms):
            self.active_waveform_index = len(self.waveforms) - 1

        return True

    def get_waveform(self, waveform_id: int) -> Optional[WaveformState]:
        """
        Get waveform by ID.

        Args:
            waveform_id: ID of waveform to retrieve

        Returns:
            WaveformState or None if not found
        """
        for waveform in self.waveforms:
            if waveform.id == waveform_id:
                return waveform
        return None

    def get_active_waveform(self) -> Optional[WaveformState]:
        """
        Get currently active waveform.

        Returns:
            Active WaveformState or None
        """
        if 0 <= self.active_waveform_index < len(self.waveforms):
            return self.waveforms[self.active_waveform_index]
        return None

    def get_enabled_waveforms(self) -> List[WaveformState]:
        """
        Get list of enabled waveforms.

        Returns:
            List of enabled WaveformState objects
        """
        return [w for w in self.waveforms if w.enabled]

    def can_show_envelopes(self) -> bool:
        """
        Check if envelopes can be shown (requires >1 enabled waveform).

        Returns:
            True if envelopes can be displayed
        """
        return len(self.get_enabled_waveforms()) > 1

    def set_time_span(self, time_span: float) -> None:
        """
        Set time span with bounds checking.

        Args:
            time_span: Time span in seconds (0.5-120.0)
        """
        self.time_span = max(0.5, min(120.0, time_span))


# Global singleton instance
app_state = AppState()
