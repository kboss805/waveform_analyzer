"""
Configuration loader for Waveform Analyzer.

Reads default settings from default.cfg (INI format).
"""

import configparser
import os
import sys
from typing import Any

CONFIG_FILENAME = "default.cfg"


def _get_config_path() -> str:
    """Return path to default.cfg.

    In PyInstaller bundles, looks next to the executable.
    In development, looks alongside the source files.
    """
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, CONFIG_FILENAME)


def load_config() -> dict[str, Any]:
    """Load configuration from default.cfg.

    Returns:
        Dict of configuration values. Missing or invalid keys fall back
        to built-in defaults.
    """
    defaults: dict[str, Any] = {
        "duration": 10.0,
        "frequency": 0.2,
        "amplitude": 2.0,
        "offset": 8.0,
        "duty_cycle": 50.0,
        "waveform_type": "sine",
        "y_axis_title": "Amplitude",
        "y_min": 0.0,
        "y_max": 10.0,
    }

    config_path = _get_config_path()
    if not os.path.exists(config_path):
        return defaults

    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")

    _read_float(parser, "global", "duration", defaults)
    _read_float(parser, "waveform_defaults", "frequency", defaults)
    _read_float(parser, "waveform_defaults", "amplitude", defaults)
    _read_float(parser, "waveform_defaults", "offset", defaults)
    _read_float(parser, "waveform_defaults", "duty_cycle", defaults)

    if parser.has_option("waveform_defaults", "type"):
        wt = parser.get("waveform_defaults", "type").strip().lower()
        if wt in ("sine", "square", "sawtooth", "triangle"):
            defaults["waveform_type"] = wt

    if parser.has_option("display", "y_axis_title"):
        defaults["y_axis_title"] = parser.get("display", "y_axis_title").strip()
    _read_float(parser, "display", "y_min", defaults)
    _read_float(parser, "display", "y_max", defaults)

    return defaults


def save_config(settings: dict[str, Any]) -> bool:
    """Write configuration to default.cfg.

    Args:
        settings: Dict with keys: duration, frequency, amplitude, offset,
                  duty_cycle, waveform_type, y_axis_title.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        config_path = _get_config_path()
        lines = [
            "# Waveform Analyzer - Default Configuration",
            "# Edit this file to customize startup defaults.",
            "# Changes take effect on next application launch.",
            "",
            "[global]",
            "# Wave duration in seconds (0.5 - 120.0)",
            f"duration = {settings.get('duration', 10.0)}",
            "",
            "[waveform_defaults]",
            "# Default waveform type: sine, square, sawtooth, triangle",
            f"type = {settings.get('waveform_type', 'sine')}",
            "# Frequency in Hz (0.1 - 100.0)",
            f"frequency = {settings.get('frequency', 0.2)}",
            "# Amplitude (0.0 - 10.0)",
            f"amplitude = {settings.get('amplitude', 2.0)}",
            "# Y-axis offset (0.0 - 10.0)",
            f"offset = {settings.get('offset', 8.0)}",
            "# Duty cycle percentage (1 - 100, Square waves only)",
            f"duty_cycle = {settings.get('duty_cycle', 50.0)}",
            "",
            "[display]",
            "# Y-axis label for the plot",
            f"y_axis_title = {settings.get('y_axis_title', 'Amplitude')}",
            "# Y-axis minimum and maximum values",
            f"y_min = {settings.get('y_min', 0.0)}",
            f"y_max = {settings.get('y_max', 10.0)}",
            "",
        ]
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return True
    except OSError:
        return False


def _read_float(
    parser: configparser.ConfigParser,
    section: str,
    key: str,
    dest: dict[str, Any]
) -> None:
    """Read a float value from config into dest, silently ignoring errors."""
    if parser.has_option(section, key):
        try:
            dest[key] = parser.getfloat(section, key)
        except ValueError:
            pass
