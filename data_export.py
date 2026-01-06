"""
CSV export functionality for waveform data.

This module handles exporting waveform data to CSV format without UI logic.
"""

import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime
import re


def sanitize_filename(filepath: str) -> str:
    """
    Sanitize filepath by removing invalid characters from the filename portion
    and ensuring .csv extension.

    Args:
        filepath: Original filepath (can be full path or just filename)

    Returns:
        Sanitized filepath with .csv extension
    """
    import os

    # Split into directory and filename
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    # Remove invalid characters from filename only (not path separators)
    # These are invalid in Windows filenames: < > : " | ? *
    filename = re.sub(r'[<>:"|?*]', '', filename)

    # Ensure .csv extension
    if not filename.lower().endswith('.csv'):
        filename += '.csv'

    # Default name if empty
    if filename == '.csv':
        filename = 'waveforms.csv'

    # Reconstruct full path if directory was provided
    if directory:
        return os.path.join(directory, filename)
    return filename


def export_to_csv(
    filename: str,
    waveforms: List[Tuple[str, np.ndarray, np.ndarray, dict]],
    envelopes: Optional[List[Tuple[str, np.ndarray, np.ndarray]]] = None,
    sample_rate: int = 1000,
    duration: float = 1.0
) -> Tuple[bool, str]:
    """
    Export waveform data to CSV file.

    Args:
        filename: Destination filename
        waveforms: List of (name, time, amplitude, params) tuples
        envelopes: Optional list of (name, time, amplitude) tuples for envelopes
        sample_rate: Sample rate in samples/second
        duration: Duration in seconds

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Sanitize filename
        filename = sanitize_filename(filename)

        # Prepare header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"# Exported: {timestamp}"]

        # Add waveform metadata
        for name, _, _, params in waveforms:
            metadata_parts = [
                f"# {name}: {params['wave_type'].capitalize()}",
                f"{params['frequency']} Hz",
                f"{params['amplitude']} amplitude"
            ]

            # Add duty cycle if applicable
            if params['wave_type'].lower() in ['square', 'pulse']:
                metadata_parts.append(f"{params['duty_cycle']}% duty cycle")

            lines.append(", ".join(metadata_parts))

        # Add envelope metadata
        if envelopes:
            num_waveforms = len(waveforms)
            for env_name, _, _ in envelopes:
                lines.append(
                    f"# {env_name}: Computed from {num_waveforms} waveforms"
                )

        # Add sample rate and duration
        lines.append(f"# Sample Rate: {sample_rate} S/s, Duration: {duration}s")

        # Create column headers
        headers = ["Time (s)"]
        for name, _, _, _ in waveforms:
            headers.append(name)

        if envelopes:
            for env_name, _, _ in envelopes:
                headers.append(env_name)

        lines.append(",".join(headers))

        # Get time array (shared time base)
        if waveforms:
            time = waveforms[0][1]
        elif envelopes:
            time = envelopes[0][1]
        else:
            return False, "No data to export"

        # Build data rows
        num_samples = len(time)
        for i in range(num_samples):
            row = [f"{time[i]:.6f}"]

            # Add waveform amplitudes
            for _, _, amplitude, _ in waveforms:
                row.append(f"{amplitude[i]:.6f}")

            # Add envelope amplitudes
            if envelopes:
                for _, _, amplitude in envelopes:
                    row.append(f"{amplitude[i]:.6f}")

            lines.append(",".join(row))

        # Write to file
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            f.write("\n".join(lines))

        return True, f"Successfully exported to {filename}"

    except PermissionError:
        return False, f"Permission denied: Cannot write to {filename}"
    except OSError as e:
        return False, f"File error: {str(e)}"
    except Exception as e:
        return False, f"Export failed: {str(e)}"


def prepare_waveform_for_export(
    name: str,
    time: np.ndarray,
    amplitude: np.ndarray,
    wave_type: str,
    frequency: float,
    amplitude_val: float,
    duty_cycle: float = 50.0
) -> Tuple[str, np.ndarray, np.ndarray, dict]:
    """
    Prepare waveform data in format suitable for export.

    Args:
        name: Waveform name
        time: Time array
        amplitude: Amplitude array
        wave_type: Type of waveform
        frequency: Frequency in Hz
        amplitude_val: Amplitude value
        duty_cycle: Duty cycle percentage (for square/pulse)

    Returns:
        Tuple of (name, time, amplitude, params_dict)
    """
    params = {
        'wave_type': wave_type,
        'frequency': frequency,
        'amplitude': amplitude_val,
        'duty_cycle': duty_cycle
    }

    return (name, time, amplitude, params)
