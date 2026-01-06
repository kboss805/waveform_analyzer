"""
Waveform Generator/Analyzer - Main Entry Point

This module initializes CustomTkinter and runs the application.
"""

from ui_components import WaveformApp


def main():
    """Initialize and run the application."""
    app = WaveformApp()
    app.mainloop()


if __name__ == "__main__":
    main()
