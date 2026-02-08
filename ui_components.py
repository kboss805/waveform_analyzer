"""
UI components for the Waveform Analyzer.

This module contains all CustomTkinter UI creation and callback logic.
"""

import os
import sys
from typing import Any, Optional, Tuple
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, Menu, Toplevel, Label
from CTkMenuBar import CTkMenuBar, CustomDropdownMenu
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure

from app_state import (
    app_state,
    DEFAULT_DURATION, DEFAULT_FREQ, DEFAULT_AMP, DEFAULT_OFFSET, DEFAULT_DUTY_CYCLE,
    DURATION_MIN, DURATION_MAX, DURATION_STEP,
    FREQ_MIN, FREQ_MAX, FREQ_STEP,
    AMP_MIN, AMP_MAX, AMP_STEP,
    OFFSET_MIN, OFFSET_MAX, OFFSET_STEP,
    DUTY_MIN, DUTY_MAX, DUTY_STEP
)
from config import load_config, save_config
from waveform_generator import gen_wf, compute_max_env, compute_min_env, compute_rms_env
from data_export import export_to_csv, prep_wf_for_export


# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Application version
APP_VERSION = "1.1.0"

# Color constants
SECTION_HEADER_COLOR = "#FFFF00"  # Yellow
ENABLED_TEXT_COLOR = "#FFFFFF"
DISABLED_TEXT_COLOR = "#808080"

# Theme colors
COLOR_DARK_BG = "#2b2b2b"
COLOR_PLOT_BG = "#1a1a1a"
COLOR_SELECTED_BG = "#3d3d6b"
COLOR_SELECTED_BORDER = "#6496ff"
COLOR_SEPARATOR = "#666666"
COLOR_BORDER = "#444444"
COLOR_WF_ON = "#009600"
COLOR_WF_OFF = "#646464"
COLOR_REMOVE_BTN = "#8B0000"
COLOR_SUCCESS = "#00FF00"
COLOR_ERROR = "#FF0000"
COLOR_RMS = "#FFA500"
COLOR_P2P_FILL = "#00FFFF"
COLOR_CURSOR_DEFAULT = "#FFFFFF"
COLOR_CURSOR_PINNED = "#AAAAAA"

# Glow effect parameters
GLOW_LINEWIDTHS = [8, 6, 4]
GLOW_ALPHAS = [0.1, 0.2, 0.3]
GLOW_CORE_WIDTH = 2

# Cursor parameters
CURSOR_PROXIMITY_THRESHOLD = 0.04  # 4% of visible Y range

# Window dimensions
WINDOW_DEFAULT_SIZE = "1200x900"
WINDOW_MIN_WIDTH = 1000
WINDOW_MIN_HEIGHT = 800
CONFIG_DIALOG_SIZE = "420x640"
ABOUT_DIALOG_SIZE = "400x310"


class WaveformApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Waveform Analyzer")
        self.geometry(WINDOW_DEFAULT_SIZE)
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Set window icon
        self._set_icon()

        # Load display settings from config (live — can be updated at runtime)
        _cfg = load_config()
        self._plot_y_min: float = _cfg["y_min"]
        self._plot_y_max: float = _cfg["y_max"]
        self._plot_y_title: str = _cfg["y_axis_title"]

        # Store widget references
        self.wf_buttons: list = []
        self.toggle_buttons: list = []
        self.remove_buttons: list = []
        self._tooltip: Optional[Any] = None

        # Cached waveform data for cursor proximity checks
        self._cached_wf_data: list[Tuple[np.ndarray, np.ndarray]] = []

        # Create UI components
        self._create_menu_bar()

        # Create content frame to hold grid-based layout (menu bar uses pack)
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True)

        # Configure grid weights for content frame
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        self._create_sidebar()
        self._create_plot_area()
        self._create_status_bar()

        # Connect cursor events (always on)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)

        # Initialize UI state
        self._update_wf_list()
        self._update_wf_parameters()
        self._update_env_controls()
        self._update_add_button()
        self._update_all_plots()

    @staticmethod
    def _get_icon_path() -> str:
        """Return icon path for both PyInstaller-bundled and dev environments."""
        if getattr(sys, 'frozen', False):
            base_path = getattr(sys, '_MEIPASS')
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, "icon.ico")

    def _set_icon(self):
        """Set the window icon if available."""
        icon_path = self._get_icon_path()
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)

    def _create_menu_bar(self):
        """Create the application menu bar using CTkMenuBar."""
        self.menu_bar = CTkMenuBar(
            master=self,
            bg_color=(COLOR_DARK_BG, COLOR_PLOT_BG),
            height=28,
            padx=5,
            pady=2
        )

        # File menu
        file_btn = self.menu_bar.add_cascade(
            text="File",
            text_color=["#ffffff", "#ffffff"],
            fg_color="transparent"
        )
        file_dropdown = CustomDropdownMenu(
            widget=file_btn,
            width=150,
            height=30,
            bg_color=(COLOR_DARK_BG, COLOR_DARK_BG),
            border_color=(COLOR_BORDER, COLOR_BORDER),
            text_color=(ENABLED_TEXT_COLOR, ENABLED_TEXT_COLOR),
            hover_color=(COLOR_SELECTED_BG, COLOR_SELECTED_BG),
            corner_radius=6
        )
        file_dropdown.add_option(
            option="Configure...",
            command=self._on_configure
        )

        # Help menu
        help_btn = self.menu_bar.add_cascade(
            text="Help",
            text_color=["#ffffff", "#ffffff"],
            fg_color="transparent"
        )
        help_dropdown = CustomDropdownMenu(
            widget=help_btn,
            width=150,
            height=30,
            bg_color=(COLOR_DARK_BG, COLOR_DARK_BG),
            border_color=(COLOR_BORDER, COLOR_BORDER),
            text_color=(ENABLED_TEXT_COLOR, ENABLED_TEXT_COLOR),
            hover_color=(COLOR_SELECTED_BG, COLOR_SELECTED_BG),
            corner_radius=6
        )
        help_dropdown.add_option(
            option="About...",
            command=self._show_about_dialog
        )

    def _on_configure(self):
        """Open the Configure dialog."""
        current = load_config()

        dialog = ctk.CTkToplevel(self)
        dialog.title("Configure Defaults")
        dialog.geometry(CONFIG_DIALOG_SIZE)
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        icon_path = self._get_icon_path()
        if os.path.exists(icon_path):
            dialog.after(200, lambda: dialog.iconbitmap(icon_path))

        dialog.update_idletasks()
        dlg_w, dlg_h = 420, 640
        x = self.winfo_x() + (self.winfo_width() - dlg_w) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg_h) // 2
        dialog.geometry(f"+{x}+{y}")

        content = ctk.CTkFrame(dialog, fg_color="transparent")
        content.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(
            content, text="Configure Defaults",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 2))
        ctk.CTkLabel(
            content, text="Changes take effect on next launch.",
            text_color=COLOR_SEPARATOR
        ).pack(pady=(0, 10))

        def add_section(text: str):
            sep = ctk.CTkFrame(content, height=1, fg_color=COLOR_SEPARATOR)
            sep.pack(fill="x", pady=(8, 4))
            ctk.CTkLabel(
                content, text=text,
                text_color=SECTION_HEADER_COLOR,
                font=ctk.CTkFont(weight="bold")
            ).pack(anchor="w")

        def add_row(label_text: str, default_val: Any) -> ctk.CTkEntry:
            row = ctk.CTkFrame(content, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=label_text, width=160, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, width=180)
            entry.insert(0, str(default_val))
            entry.pack(side="left")
            return entry

        # Global
        add_section("Global")
        dur_entry = add_row("Duration (s):", current["duration"])

        # Waveform Defaults
        add_section("Waveform Defaults")
        type_var = ctk.StringVar(value=current["waveform_type"])
        type_row = ctk.CTkFrame(content, fg_color="transparent")
        type_row.pack(fill="x", pady=2)
        ctk.CTkLabel(type_row, text="Type:", width=160, anchor="w").pack(side="left")
        ctk.CTkOptionMenu(
            type_row, variable=type_var,
            values=["sine", "square", "sawtooth", "triangle"], width=180
        ).pack(side="left")
        freq_entry = add_row("Frequency (Hz):", current["frequency"])
        amp_entry = add_row("Amplitude:", current["amplitude"])
        offset_entry = add_row("Offset:", current["offset"])
        duty_entry = add_row("Duty Cycle (%) - For Square Waves:", current["duty_cycle"])

        # Display
        add_section("Display")
        y_title_entry = add_row("Y-Axis Title:", current["y_axis_title"])
        y_min_entry = add_row("Y-Axis Min:", current["y_min"])
        y_max_entry = add_row("Y-Axis Max:", current["y_max"])

        # Status label
        status_lbl = ctk.CTkLabel(content, text="")
        status_lbl.pack(pady=(10, 0))

        # Buttons
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(5, 0))

        def on_save():
            try:
                new_settings = {
                    "duration": max(DURATION_MIN, min(DURATION_MAX, float(dur_entry.get()))),
                    "waveform_type": type_var.get(),
                    "frequency": max(FREQ_MIN, min(FREQ_MAX, float(freq_entry.get()))),
                    "amplitude": max(AMP_MIN, min(AMP_MAX, float(amp_entry.get()))),
                    "offset": max(OFFSET_MIN, min(OFFSET_MAX, float(offset_entry.get()))),
                    "duty_cycle": max(DUTY_MIN, min(DUTY_MAX, float(duty_entry.get()))),
                    "y_axis_title": y_title_entry.get().strip() or "Amplitude",
                    "y_min": float(y_min_entry.get()),
                    "y_max": float(y_max_entry.get()),
                }
            except ValueError:
                status_lbl.configure(
                    text="Invalid value. Please check inputs.",
                    text_color=COLOR_ERROR
                )
                return
            if new_settings["y_min"] >= new_settings["y_max"]:
                status_lbl.configure(
                    text="Y-Axis Min must be less than Max.",
                    text_color=COLOR_ERROR
                )
                return
            if save_config(new_settings):
                # Apply display settings immediately
                self._plot_y_min = new_settings["y_min"]
                self._plot_y_max = new_settings["y_max"]
                self._plot_y_title = new_settings["y_axis_title"]
                self._update_all_plots()
                status_lbl.configure(
                    text="Saved. Waveform defaults apply on next launch.",
                    text_color=COLOR_SUCCESS
                )
            else:
                status_lbl.configure(
                    text="Failed to save configuration.",
                    text_color=COLOR_ERROR
                )

        ctk.CTkButton(btn_frame, text="Save", width=100, command=on_save).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Cancel", width=100, command=dialog.destroy).pack(side="left")

    def _show_about_dialog(self):
        """Show the About dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("About Waveform Analyzer")
        dialog.geometry(ABOUT_DIALOG_SIZE)
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        # Set dialog icon
        icon_path = self._get_icon_path()
        if os.path.exists(icon_path):
            dialog.after(200, lambda: dialog.iconbitmap(icon_path))

        # Center the dialog on the main window
        dialog.update_idletasks()
        dlg_w, dlg_h = 400, 310
        x = self.winfo_x() + (self.winfo_width() - dlg_w) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg_h) // 2
        dialog.geometry(f"+{x}+{y}")

        # Content frame
        content = ctk.CTkFrame(dialog, fg_color="transparent")
        content.pack(expand=True, fill="both", padx=20, pady=20)

        # Title
        ctk.CTkLabel(
            content, text="Waveform Analyzer",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 10))

        # Version
        ctk.CTkLabel(content, text=f"Version {APP_VERSION}").pack()

        # Description
        ctk.CTkLabel(
            content,
            text="Waveform Analyzer is a tool to help engineers\nand scientists visualize different waveforms\nin real-time.",
            justify="center"
        ).pack(pady=15)

        # Author
        ctk.CTkLabel(
            content, text="Kevin Bossoletti",
            text_color="#aaaaaa"
        ).pack(pady=(0, 0))
        ctk.CTkLabel(
            content, text="kevin.bossoletti@gmail.com",
            text_color="#888888", height=20
        ).pack(pady=(0, 0))

        # Close button
        ctk.CTkButton(
            content, text="OK", width=80,
            command=dialog.destroy
        ).pack(pady=(20, 0))

    def _create_sidebar(self):
        """Create the sidebar with all controls."""
        # Sidebar frame with scrolling
        self.sidebar = ctk.CTkScrollableFrame(self.content_frame, width=330)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 0))

        # === Waveforms ===
        self._add_section_header("Waveforms")

        # Add Waveform button
        self.add_wf_btn = ctk.CTkButton(
            self.sidebar, text="+ Add Waveform",
            command=self._on_add_wf
        )
        self.add_wf_btn.pack(fill="x", pady=(5, 5))

        # Waveform list container
        self.wf_list_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.wf_list_frame.pack(fill="x", pady=(0, 10))

        # === Waveform Parameters ===
        self._add_section_header("Waveform Parameters")

        # Waveform Type
        ctk.CTkLabel(self.sidebar, text="Type").pack(anchor="w", pady=(5, 2))
        self.wf_type_combo = ctk.CTkComboBox(
            self.sidebar,
            values=["Sine", "Square", "Sawtooth", "Triangle"],
            command=self._on_wf_type_changed,
            width=200
        )
        self.wf_type_combo.pack(anchor="w", pady=(0, 5))
        self.wf_type_combo.set("Sine")

        # Wave Duration
        _, self.duration_frame, self.duration_entry, \
            self.duration_dec_btn, self.duration_inc_btn = \
            self._create_param_row(
                "Wave Duration (s)", DEFAULT_DURATION,
                self._on_duration_enter,
                self._on_duration_dec, self._on_duration_inc
            )

        # Offset
        _, self.offset_frame, self.offset_entry, \
            self.offset_dec_btn, self.offset_inc_btn = \
            self._create_param_row(
                "Offset", DEFAULT_OFFSET,
                self._on_offset_enter,
                self._on_offset_dec, self._on_offset_inc
            )

        # Frequency
        _, self.freq_frame, self.freq_entry, \
            self.freq_dec_btn, self.freq_inc_btn = \
            self._create_param_row(
                "Frequency (Hz)", DEFAULT_FREQ,
                self._on_freq_enter,
                self._on_freq_dec, self._on_freq_inc
            )

        # Amplitude
        _, self.amp_frame, self.amp_entry, \
            self.amp_dec_btn, self.amp_inc_btn = \
            self._create_param_row(
                "Amplitude", DEFAULT_AMP,
                self._on_amp_enter,
                self._on_amp_dec, self._on_amp_inc
            )

        # Duty Cycle (hidden by default, shown for Square waves)
        self.duty_label, self.duty_frame, self.duty_entry, \
            self.duty_dec_btn, self.duty_inc_btn = \
            self._create_param_row(
                "Duty Cycle (%)", DEFAULT_DUTY_CYCLE,
                self._on_duty_enter,
                self._on_duty_dec, self._on_duty_inc,
                pack=False
            )

        # === Advanced ===
        self._add_section_header("Advanced")

        # Max Envelope checkbox
        max_env_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        max_env_frame.pack(fill="x", pady=2)
        self.show_max_env_var = ctk.BooleanVar(value=False)
        self.show_max_env_cb = ctk.CTkCheckBox(
            max_env_frame, text="",
            variable=self.show_max_env_var,
            command=self._on_max_env_changed,
            width=24
        )
        self.show_max_env_cb.pack(side="left")
        self.show_max_env_label = ctk.CTkLabel(max_env_frame, text="Show Max Envelope")
        self.show_max_env_label.pack(side="left")

        # Min Envelope checkbox
        min_env_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        min_env_frame.pack(fill="x", pady=2)
        self.show_min_env_var = ctk.BooleanVar(value=False)
        self.show_min_env_cb = ctk.CTkCheckBox(
            min_env_frame, text="",
            variable=self.show_min_env_var,
            command=self._on_min_env_changed,
            width=24
        )
        self.show_min_env_cb.pack(side="left")
        self.show_min_env_label = ctk.CTkLabel(min_env_frame, text="Show Min Envelope")
        self.show_min_env_label.pack(side="left")

        # RMS Envelope checkbox
        rms_env_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        rms_env_frame.pack(fill="x", pady=2)
        self.show_rms_env_var = ctk.BooleanVar(value=False)
        self.show_rms_env_cb = ctk.CTkCheckBox(
            rms_env_frame, text="",
            variable=self.show_rms_env_var,
            command=self._on_rms_env_changed,
            width=24
        )
        self.show_rms_env_cb.pack(side="left")
        self.show_rms_env_label = ctk.CTkLabel(rms_env_frame, text="Show RMS Envelope")
        self.show_rms_env_label.pack(side="left")

        # Live cursor state: tracks mouse, click pins a reference
        self._live_cursor_x: Optional[float] = None
        self._live_cursor_y: Optional[float] = None
        self._live_cursor_vline: Optional[Any] = None
        self._pinned_cursor_x: Optional[float] = None
        self._pinned_cursor_vline: Optional[Any] = None
        self._highlight_marker: Optional[Any] = None
        self._highlighted_wf_name: Optional[str] = None

        # === Export ===
        self._add_section_header("Export")

        ctk.CTkButton(
            self.sidebar, text="Export to CSV",
            command=self._on_export_clicked
        ).pack(fill="x", pady=(5, 5))

        self.export_status = ctk.CTkLabel(
            self.sidebar, text="Status: Ready",
            text_color=COLOR_SUCCESS
        )
        self.export_status.pack(anchor="w", pady=(5, 10))

    def _create_plot_area(self):
        """Create the matplotlib plot area."""
        # Plot container
        self.plot_frame = ctk.CTkFrame(self.content_frame)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 0))
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

        # Create matplotlib figure with dark theme
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(8, 6), facecolor=COLOR_PLOT_BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLOR_PLOT_BG)

        # Configure axes
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(self._plot_y_title)
        self.ax.grid(True, alpha=0.3, color=COLOR_SEPARATOR)

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Add navigation toolbar for pan/zoom
        toolbar_frame = ctk.CTkFrame(self.plot_frame, fg_color="transparent")
        toolbar_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = ctk.CTkLabel(
            self.content_frame, text="Waveforms: 1/5",
            anchor="w"
        )
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))

    def _add_section_header(self, text: str):
        """Add a yellow section header with a separator line above it."""
        separator = ctk.CTkFrame(self.sidebar, height=1, fg_color=COLOR_SEPARATOR)
        separator.pack(fill="x", pady=(10, 5))
        header = ctk.CTkLabel(
            self.sidebar, text=text,
            text_color=SECTION_HEADER_COLOR,
            font=ctk.CTkFont(weight="bold")
        )
        header.pack(anchor="w")

    def _create_param_row(
        self,
        label_text: str,
        default_val: float,
        on_enter: Any,
        on_dec: Any,
        on_inc: Any,
        pack: bool = True
    ) -> Tuple[ctk.CTkLabel, ctk.CTkFrame, ctk.CTkEntry, ctk.CTkButton, ctk.CTkButton]:
        """Create a parameter input row with label, entry, and +/- buttons.

        Args:
            label_text: Text for the row label.
            default_val: Initial entry value.
            on_enter: Callback for Return/FocusOut on the entry.
            on_dec: Callback for the minus button.
            on_inc: Callback for the plus button.
            pack: Whether to pack the label and frame immediately.

        Returns:
            Tuple of (label, frame, entry, dec_btn, inc_btn).
        """
        label = ctk.CTkLabel(self.sidebar, text=label_text)
        frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")

        entry = ctk.CTkEntry(frame, width=120)
        entry.pack(side="left", padx=(0, 5))
        entry.insert(0, f"{default_val:.1f}")
        entry.bind("<Return>", on_enter)
        entry.bind("<FocusOut>", on_enter)

        dec_btn = ctk.CTkButton(frame, text="-", width=30, command=on_dec)
        dec_btn.pack(side="left", padx=2)

        inc_btn = ctk.CTkButton(frame, text="+", width=30, command=on_inc)
        inc_btn.pack(side="left", padx=2)

        if pack:
            label.pack(anchor="w", pady=(5, 2))
            frame.pack(fill="x", pady=(0, 5))

        return label, frame, entry, dec_btn, inc_btn

    # === Callback Methods ===

    def _is_duplicate_name(self, name: str, exclude_id: int) -> bool:
        """Check if a waveform name is already in use."""
        for wf in app_state.wfs:
            if wf.id != exclude_id and wf.display_name == name:
                return True
        return False

    def _on_rename_wf(self, wf_id: int):
        """Show rename dialog for a waveform."""
        wf = app_state.get_wf(wf_id)
        if not wf:
            return

        prompt = f"Enter new name for {wf.display_name}:"
        while True:
            dialog = ctk.CTkInputDialog(
                text=prompt,
                title="Rename Waveform"
            )
            new_name = dialog.get_input()

            if new_name is None:
                return  # User cancelled

            new_name = new_name.strip()

            # Empty name reverts to default - check that default isn't taken
            check_name = new_name if new_name else f"Waveform {wf.id + 1}"
            if not self._is_duplicate_name(check_name, wf_id):
                wf.name = new_name
                self._update_wf_list()
                self._update_all_plots()
                return

            prompt = f'"{check_name}" is already in use.\nEnter a different name:'

    def _show_wf_context_menu(self, event: tk.Event, wf_id: int):
        """Show right-click context menu for a waveform button."""
        menu_style = {
            "bg": COLOR_DARK_BG,
            "fg": ENABLED_TEXT_COLOR,
            "activebackground": COLOR_SELECTED_BG,
            "activeforeground": ENABLED_TEXT_COLOR,
            "relief": "flat",
            "borderwidth": 0,
        }
        ctx_menu = Menu(self, tearoff=0, **menu_style)
        ctx_menu.add_command(
            label="Rename...",
            command=lambda: self._on_rename_wf(wf_id)
        )
        ctx_menu.bind("<Unmap>", lambda _: ctx_menu.destroy())
        ctx_menu.tk_popup(event.x_root, event.y_root)

    def _show_tooltip(self, event: tk.Event):
        """Show tooltip near the cursor."""
        self._hide_tooltip()
        tip = Toplevel(self)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
        lbl = Label(
            tip, text="Right-click to change waveform name",
            background=COLOR_DARK_BG, foreground=ENABLED_TEXT_COLOR,
            relief="solid", borderwidth=1,
            padx=6, pady=3, font=("Segoe UI", 9)
        )
        lbl.pack()
        self._tooltip = tip

    def _hide_tooltip(self, event: Optional[tk.Event] = None):
        """Destroy the tooltip if it exists."""
        if self._tooltip:
            self._tooltip.destroy()
            self._tooltip = None

    def _on_duration_enter(self, event: Optional[tk.Event] = None):
        """Handle duration entry."""
        try:
            value = float(self.duration_entry.get())
            value = max(DURATION_MIN, min(DURATION_MAX, value))
            app_state.set_duration(value)
            self.duration_entry.delete(0, "end")
            self.duration_entry.insert(0, f"{value:.1f}")
            self._update_duration_btns()
            self._update_all_plots()
        except ValueError:
            self.duration_entry.delete(0, "end")
            self.duration_entry.insert(0, f"{app_state.duration:.1f}")

    def _on_duration_inc(self):
        """Increment duration."""
        new_value = min(DURATION_MAX, app_state.duration + DURATION_STEP)
        app_state.set_duration(new_value)
        self.duration_entry.delete(0, "end")
        self.duration_entry.insert(0, f"{new_value:.1f}")
        self._update_duration_btns()
        self._update_all_plots()

    def _on_duration_dec(self):
        """Decrement duration."""
        new_value = max(DURATION_MIN, app_state.duration - DURATION_STEP)
        app_state.set_duration(new_value)
        self.duration_entry.delete(0, "end")
        self.duration_entry.insert(0, f"{new_value:.1f}")
        self._update_duration_btns()
        self._update_all_plots()

    def _on_max_env_changed(self):
        """Handle max envelope toggle."""
        app_state.show_max_env = self.show_max_env_var.get()
        self._auto_hide_source_waveforms()
        self._update_env_controls()
        self._update_all_plots()

    def _on_min_env_changed(self):
        """Handle min envelope toggle."""
        app_state.show_min_env = self.show_min_env_var.get()
        self._auto_hide_source_waveforms()
        self._update_env_controls()
        self._update_all_plots()

    def _on_rms_env_changed(self):
        """Handle RMS envelope toggle."""
        app_state.show_rms_env = self.show_rms_env_var.get()
        self._auto_hide_source_waveforms()
        self._update_env_controls()
        self._update_all_plots()

    def _auto_hide_source_waveforms(self):
        """Automatically hide/show source waveforms based on envelope state."""
        any_envelope_shown = (
            app_state.show_max_env
            or app_state.show_min_env
            or app_state.show_rms_env
        )
        app_state.hide_src_wfs = any_envelope_shown
        self._update_wf_management_controls()

    def _on_add_wf(self):
        """Add a new waveform."""
        new_wf = app_state.add_wf()
        if new_wf:
            self._update_wf_list()
            self._update_wf_parameters()
            self._update_env_controls()
            self._update_all_plots()
            self._update_add_button()

    def _on_remove_wf(self, wf_id: int):
        """Remove a waveform."""
        if app_state.remove_wf(wf_id):
            self._update_wf_list()
            self._update_wf_parameters()
            self._update_env_controls()
            self._update_all_plots()
            self._update_add_button()

    def _on_toggle_wf(self, wf_id: int):
        """Toggle waveform visibility."""
        wf = app_state.get_wf(wf_id)
        if wf:
            wf.enabled = not wf.enabled
            self._update_env_controls()
            self._update_all_plots()
            self._update_wf_list()

    def _on_select_wf(self, wf_id: int):
        """Select a waveform for editing."""
        app_state.active_wf_index = wf_id
        self._update_wf_parameters()
        self._update_wf_list()

    def _on_wf_type_changed(self, value: str):
        """Handle waveform type change."""
        wf = app_state.get_active_wf()
        if wf:
            wf.wf_type = value.lower()
            self._update_wf_parameters()
            self._update_all_plots()

    def _on_freq_enter(self, event: Optional[tk.Event] = None):
        """Handle frequency entry."""
        wf = app_state.get_active_wf()
        if wf:
            try:
                value = float(self.freq_entry.get())
                value = max(FREQ_MIN, min(FREQ_MAX, value))
                wf.freq = value
                self.freq_entry.delete(0, "end")
                self.freq_entry.insert(0, f"{value:.1f}")
                self._update_freq_btns()
                self._update_all_plots()
            except ValueError:
                self.freq_entry.delete(0, "end")
                self.freq_entry.insert(0, f"{wf.freq:.1f}")

    def _on_freq_inc(self):
        """Increment frequency."""
        wf = app_state.get_active_wf()
        if wf:
            new_value = min(FREQ_MAX, wf.freq + FREQ_STEP)
            wf.freq = new_value
            self.freq_entry.delete(0, "end")
            self.freq_entry.insert(0, f"{new_value:.1f}")
            self._update_freq_btns()
            self._update_all_plots()

    def _on_freq_dec(self):
        """Decrement frequency."""
        wf = app_state.get_active_wf()
        if wf:
            new_value = max(FREQ_MIN, wf.freq - FREQ_STEP)
            wf.freq = new_value
            self.freq_entry.delete(0, "end")
            self.freq_entry.insert(0, f"{new_value:.1f}")
            self._update_freq_btns()
            self._update_all_plots()

    def _on_amp_enter(self, event: Optional[tk.Event] = None):
        """Handle amplitude entry."""
        wf = app_state.get_active_wf()
        if wf:
            try:
                value = float(self.amp_entry.get())
                value = max(AMP_MIN, min(AMP_MAX, value))
                wf.amp = value
                self.amp_entry.delete(0, "end")
                self.amp_entry.insert(0, f"{value:.1f}")
                self._update_amp_btns()
                self._update_all_plots()
            except ValueError:
                self.amp_entry.delete(0, "end")
                self.amp_entry.insert(0, f"{wf.amp:.1f}")

    def _on_amp_inc(self):
        """Increment amplitude."""
        wf = app_state.get_active_wf()
        if wf:
            new_value = min(AMP_MAX, wf.amp + AMP_STEP)
            wf.amp = new_value
            self.amp_entry.delete(0, "end")
            self.amp_entry.insert(0, f"{new_value:.1f}")
            self._update_amp_btns()
            self._update_all_plots()

    def _on_amp_dec(self):
        """Decrement amplitude."""
        wf = app_state.get_active_wf()
        if wf:
            new_value = max(AMP_MIN, wf.amp - AMP_STEP)
            wf.amp = new_value
            self.amp_entry.delete(0, "end")
            self.amp_entry.insert(0, f"{new_value:.1f}")
            self._update_amp_btns()
            self._update_all_plots()

    def _on_offset_enter(self, event: Optional[tk.Event] = None):
        """Handle offset entry."""
        wf = app_state.get_active_wf()
        if wf:
            try:
                value = float(self.offset_entry.get())
                value = max(OFFSET_MIN, min(OFFSET_MAX, value))
                wf.offset = value
                self.offset_entry.delete(0, "end")
                self.offset_entry.insert(0, f"{value:.1f}")
                self._update_offset_btns()
                self._update_all_plots()
            except ValueError:
                self.offset_entry.delete(0, "end")
                self.offset_entry.insert(0, f"{wf.offset:.1f}")

    def _on_offset_inc(self):
        """Increment offset."""
        wf = app_state.get_active_wf()
        if wf:
            new_value = min(OFFSET_MAX, wf.offset + OFFSET_STEP)
            wf.offset = new_value
            self.offset_entry.delete(0, "end")
            self.offset_entry.insert(0, f"{new_value:.1f}")
            self._update_offset_btns()
            self._update_all_plots()

    def _on_offset_dec(self):
        """Decrement offset."""
        wf = app_state.get_active_wf()
        if wf:
            new_value = max(OFFSET_MIN, wf.offset - OFFSET_STEP)
            wf.offset = new_value
            self.offset_entry.delete(0, "end")
            self.offset_entry.insert(0, f"{new_value:.1f}")
            self._update_offset_btns()
            self._update_all_plots()

    def _on_duty_enter(self, event: Optional[tk.Event] = None):
        """Handle duty cycle entry."""
        wf = app_state.get_active_wf()
        if wf:
            try:
                value = float(self.duty_entry.get())
                value = max(DUTY_MIN, min(DUTY_MAX, value))
                wf.duty_cycle = value
                self.duty_entry.delete(0, "end")
                self.duty_entry.insert(0, f"{value:.1f}")
                self._update_duty_btns()
                self._update_all_plots()
            except ValueError:
                self.duty_entry.delete(0, "end")
                self.duty_entry.insert(0, f"{wf.duty_cycle:.1f}")

    def _on_duty_inc(self):
        """Increment duty cycle."""
        wf = app_state.get_active_wf()
        if wf:
            new_value = min(DUTY_MAX, wf.duty_cycle + DUTY_STEP)
            wf.duty_cycle = new_value
            self.duty_entry.delete(0, "end")
            self.duty_entry.insert(0, f"{new_value:.1f}")
            self._update_duty_btns()
            self._update_all_plots()

    def _on_duty_dec(self):
        """Decrement duty cycle."""
        wf = app_state.get_active_wf()
        if wf:
            new_value = max(DUTY_MIN, wf.duty_cycle - DUTY_STEP)
            wf.duty_cycle = new_value
            self.duty_entry.delete(0, "end")
            self.duty_entry.insert(0, f"{new_value:.1f}")
            self._update_duty_btns()
            self._update_all_plots()

    def _on_export_clicked(self):
        """Handle export button click - shows native file dialog."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="waveforms.csv",
            title="Export Waveforms to CSV"
        )

        if not filename:
            return  # User cancelled

        # Collect enabled waveform data
        wfs_to_export = []
        wf_arrays = []
        for wf in app_state.get_enabled_wfs():
            time, amp = gen_wf(
                wf.wf_type,
                wf.freq,
                wf.amp,
                wf.offset,
                wf.duty_cycle,
                app_state.duration,
                app_state.sample_rate
            )
            wf_arrays.append((time, amp))

            name = wf.display_name.replace(" ", "_")
            export_data = prep_wf_for_export(
                name, time, amp,
                wf.wf_type,
                wf.freq,
                wf.amp,
                wf.offset,
                wf.duty_cycle
            )
            wfs_to_export.append(export_data)

        # Collect envelope data if enabled
        envs_to_export = []
        if app_state.can_show_envelopes() and wf_arrays:
            if app_state.show_max_env:
                time, max_env = compute_max_env(wf_arrays)
                envs_to_export.append(("Max_Envelope", time, max_env))

            if app_state.show_min_env:
                time, min_env = compute_min_env(wf_arrays)
                envs_to_export.append(("Min_Envelope", time, min_env))

            if app_state.show_rms_env:
                time, rms_env = compute_rms_env(wf_arrays)
                envs_to_export.append(("RMS_Envelope", time, rms_env))

        # Export
        success, message = export_to_csv(
            filename,
            wfs_to_export,
            envs_to_export if envs_to_export else None,
            app_state.sample_rate,
            app_state.duration
        )

        # Update status
        if success:
            self.export_status.configure(text=message, text_color=COLOR_SUCCESS)
        else:
            self.export_status.configure(text=message, text_color=COLOR_ERROR)

    # === UI Update Methods ===

    def _update_all_plots(self):
        """Regenerate and update all waveform plots."""
        self.ax.clear()

        # Configure axes
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(self._plot_y_title)
        self.ax.set_xlim(0, app_state.duration)
        self.ax.set_ylim(self._plot_y_min, self._plot_y_max)
        self.ax.grid(visible=True, alpha=0.3, color=COLOR_SEPARATOR)

        # Generate and plot enabled waveforms
        wf_data: list[Tuple[np.ndarray, np.ndarray]] = []
        for wf in app_state.wfs:
            if wf.enabled:
                time, amp = gen_wf(
                    wf.wf_type,
                    wf.freq,
                    wf.amp,
                    wf.offset,
                    wf.duty_cycle,
                    app_state.duration,
                    app_state.sample_rate
                )
                wf_data.append((time, amp))

                # Only plot if not hiding source waveforms
                if not app_state.hide_src_wfs:
                    # Convert RGB tuple to matplotlib color format
                    color = tuple(c / 255 for c in wf.color)
                    self.ax.plot(
                        time, amp, color=color,
                        label=wf.display_name, linewidth=2
                    )

        # Cache waveform data for cursor proximity checks
        self._cached_wf_data = wf_data

        # Plot envelopes with glow effect
        if app_state.can_show_envelopes() and wf_data:
            self._plot_envelopes(wf_data)

        # Add legend if there are any lines
        if self.ax.get_lines():
            self.ax.legend(loc='upper right')

        # Redraw cursors (ax.clear removes them)
        self._redraw_cursors()

        # Redraw canvas
        self.canvas.draw()

        # Update status bar
        self._update_status_bar()

    def _plot_envelopes(self, wf_data: list) -> None:
        """Plot all enabled envelope lines with glow effects and P2P fill."""
        max_env_data = None
        min_env_data = None

        if app_state.show_max_env:
            max_env_data = compute_max_env(wf_data)
            self._plot_glowing_line(
                max_env_data[0], max_env_data[1],
                COLOR_SUCCESS, 'Max Envelope'
            )

        if app_state.show_min_env:
            min_env_data = compute_min_env(wf_data)
            self._plot_glowing_line(
                min_env_data[0], min_env_data[1],
                COLOR_ERROR, 'Min Envelope'
            )

        # Peak-to-Peak fill between max and min
        if max_env_data is not None and min_env_data is not None:
            self.ax.fill_between(
                max_env_data[0], min_env_data[1], max_env_data[1],
                alpha=0.12, color=COLOR_P2P_FILL, label="Peak-to-Peak"
            )

        if app_state.show_rms_env:
            time_rms, rms_env = compute_rms_env(wf_data)
            self._plot_glowing_line(
                time_rms, rms_env, COLOR_RMS, 'RMS Envelope'
            )

    def _plot_glowing_line(self, x: Any, y: Any, color: str, label: str):
        """Plot a line with a glow effect using layered transparency."""
        for lw, alpha in zip(GLOW_LINEWIDTHS, GLOW_ALPHAS):
            self.ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)
        self.ax.plot(
            x, y, color=color, linewidth=GLOW_CORE_WIDTH,
            alpha=1.0, label=label
        )

    def _update_wf_list(self):
        """Update the waveform list UI."""
        # Clear existing widgets safely
        children = list(self.wf_list_frame.winfo_children())
        for widget in children:
            try:
                widget.destroy()
            except Exception:
                pass

        self.wf_buttons = []
        self.toggle_buttons = []
        self.remove_buttons = []

        for wf in app_state.wfs:
            row_frame = ctk.CTkFrame(self.wf_list_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)

            # Selection button
            is_selected = wf.id == app_state.active_wf_index
            fg_color = COLOR_SELECTED_BG if is_selected else COLOR_DARK_BG
            border_color = COLOR_SELECTED_BORDER if is_selected else COLOR_DARK_BG
            border_width = 2 if is_selected else 0

            wf_btn = ctk.CTkButton(
                row_frame,
                text=wf.display_name,
                width=180,
                fg_color=fg_color,
                border_color=border_color,
                border_width=border_width,
                command=lambda wid=wf.id: self._on_select_wf(wid)
            )
            wf_btn.pack(side="left", padx=(0, 5))
            self.wf_buttons.append(wf_btn)

            # Right-click context menu for renaming
            wf_btn.bind(
                "<Button-3>",
                lambda e, wid=wf.id: self._show_wf_context_menu(e, wid)
            )

            # Hover tooltip
            wf_btn.bind("<Enter>", self._show_tooltip)
            wf_btn.bind("<Leave>", self._hide_tooltip)

            # Visibility toggle button
            vis_text = "ON" if wf.enabled else "OFF"
            vis_color = COLOR_WF_ON if wf.enabled else COLOR_WF_OFF
            vis_btn = ctk.CTkButton(
                row_frame,
                text=vis_text,
                width=40,
                fg_color=vis_color,
                command=lambda wid=wf.id: self._on_toggle_wf(wid)
            )
            vis_btn.pack(side="left", padx=2)
            self.toggle_buttons.append(vis_btn)

            # Remove button (only show if more than 1 waveform)
            if len(app_state.wfs) > app_state.MIN_WFS:
                is_enabled = not app_state.hide_src_wfs
                remove_btn = ctk.CTkButton(
                    row_frame,
                    text="X",
                    width=30,
                    fg_color=COLOR_REMOVE_BTN if is_enabled else COLOR_WF_OFF,
                    state="normal" if is_enabled else "disabled",
                    command=lambda wid=wf.id: self._on_remove_wf(wid)
                )
                remove_btn.pack(side="left", padx=2)
                self.remove_buttons.append(remove_btn)

    def _update_wf_parameters(self):
        """Update waveform parameter inputs based on active waveform."""
        wf = app_state.get_active_wf()
        if not wf:
            return

        # Update entry fields
        self.freq_entry.delete(0, "end")
        self.freq_entry.insert(0, f"{wf.freq:.1f}")

        self.amp_entry.delete(0, "end")
        self.amp_entry.insert(0, f"{wf.amp:.1f}")

        self.offset_entry.delete(0, "end")
        self.offset_entry.insert(0, f"{wf.offset:.1f}")

        self.duty_entry.delete(0, "end")
        self.duty_entry.insert(0, f"{wf.duty_cycle:.1f}")

        self.wf_type_combo.set(wf.wf_type.capitalize())

        # Update button states
        self._update_freq_btns()
        self._update_amp_btns()
        self._update_offset_btns()
        self._update_duty_btns()

        # Show/hide duty cycle for square waves
        needs_duty = wf.wf_type.lower() == 'square'
        if needs_duty:
            self.duty_label.pack(anchor="w", pady=(5, 2), after=self.amp_frame)
            self.duty_frame.pack(fill="x", pady=(0, 5), after=self.duty_label)
        else:
            self.duty_label.pack_forget()
            self.duty_frame.pack_forget()

    def _update_duration_btns(self):
        """Update duration button states."""
        at_min = app_state.duration <= DURATION_MIN
        at_max = app_state.duration >= DURATION_MAX
        self.duration_dec_btn.configure(state="disabled" if at_min else "normal")
        self.duration_inc_btn.configure(state="disabled" if at_max else "normal")

    def _update_freq_btns(self):
        """Update frequency button states."""
        wf = app_state.get_active_wf()
        if wf:
            at_min = wf.freq <= FREQ_MIN
            at_max = wf.freq >= FREQ_MAX
            self.freq_dec_btn.configure(state="disabled" if at_min else "normal")
            self.freq_inc_btn.configure(state="disabled" if at_max else "normal")

    def _update_amp_btns(self):
        """Update amplitude button states."""
        wf = app_state.get_active_wf()
        if wf:
            at_min = wf.amp <= AMP_MIN
            at_max = wf.amp >= AMP_MAX
            self.amp_dec_btn.configure(state="disabled" if at_min else "normal")
            self.amp_inc_btn.configure(state="disabled" if at_max else "normal")

    def _update_offset_btns(self):
        """Update offset button states."""
        wf = app_state.get_active_wf()
        if wf:
            at_min = wf.offset <= OFFSET_MIN
            at_max = wf.offset >= OFFSET_MAX
            self.offset_dec_btn.configure(state="disabled" if at_min else "normal")
            self.offset_inc_btn.configure(state="disabled" if at_max else "normal")

    def _update_duty_btns(self):
        """Update duty cycle button states."""
        wf = app_state.get_active_wf()
        if wf:
            at_min = wf.duty_cycle <= DUTY_MIN
            at_max = wf.duty_cycle >= DUTY_MAX
            self.duty_dec_btn.configure(state="disabled" if at_min else "normal")
            self.duty_inc_btn.configure(state="disabled" if at_max else "normal")

    def _update_env_controls(self):
        """Enable/disable envelope checkboxes based on number of enabled waveforms."""
        can_show = app_state.can_show_envelopes()

        # Update max envelope checkbox
        self.show_max_env_cb.configure(state="normal" if can_show else "disabled")
        self.show_max_env_label.configure(
            text_color=ENABLED_TEXT_COLOR if can_show else DISABLED_TEXT_COLOR
        )

        # Update min envelope checkbox
        self.show_min_env_cb.configure(state="normal" if can_show else "disabled")
        self.show_min_env_label.configure(
            text_color=ENABLED_TEXT_COLOR if can_show else DISABLED_TEXT_COLOR
        )

        # Update RMS envelope checkbox
        self.show_rms_env_cb.configure(state="normal" if can_show else "disabled")
        self.show_rms_env_label.configure(
            text_color=ENABLED_TEXT_COLOR if can_show else DISABLED_TEXT_COLOR
        )

        if not can_show:
            app_state.show_max_env = False
            app_state.show_min_env = False
            app_state.show_rms_env = False
            app_state.hide_src_wfs = False
            self.show_max_env_var.set(False)
            self.show_min_env_var.set(False)
            self.show_rms_env_var.set(False)

    def _update_add_button(self):
        """Enable/disable add waveform button based on max limit and hide_src state."""
        can_add = len(app_state.wfs) < app_state.MAX_WFS and not app_state.hide_src_wfs
        self.add_wf_btn.configure(state="normal" if can_add else "disabled")

    def _update_wf_management_controls(self):
        """Enable/disable waveform management controls based on hide_src state."""
        self._update_add_button()
        self._update_wf_list()

    # === Cursor Methods ===

    def _on_mouse_move(self, event: Any):
        """Handle mouse movement over the plot for live cursor tracking."""
        if event.inaxes != self.ax:
            # Remove live cursor when mouse leaves plot
            if self._live_cursor_vline and self._live_cursor_vline in self.ax.lines:
                self._live_cursor_vline.remove()
                self._live_cursor_vline = None
            self._remove_highlight_marker()
            self._live_cursor_x = None
            self._live_cursor_y = None
            self._highlighted_wf_name = None
            self.canvas.draw_idle()
            return

        self._live_cursor_x = event.xdata
        self._live_cursor_y = event.ydata

        # Find nearest waveform for highlight
        nearest = self._find_nearest_wf(event.xdata, event.ydata)
        cursor_color = COLOR_CURSOR_DEFAULT
        cursor_alpha = 0.5
        cursor_width = 1

        if nearest is not None:
            wf_name, wf_y, wf_color = nearest
            self._highlighted_wf_name = wf_name
            cursor_color = wf_color
            cursor_alpha = 0.8
            cursor_width = 1.5
            # Update or create highlight marker dot
            self._remove_highlight_marker()
            self._highlight_marker = self.ax.plot(
                event.xdata, wf_y, 'o',
                color=wf_color, markersize=8,
                markeredgecolor='white', markeredgewidth=1.5,
                zorder=10
            )[0]
        else:
            self._highlighted_wf_name = None
            self._remove_highlight_marker()

        # Update or create live cursor line
        if self._live_cursor_vline and self._live_cursor_vline in self.ax.lines:
            self._live_cursor_vline.set_xdata([event.xdata, event.xdata])
            self._live_cursor_vline.set_color(cursor_color)
            self._live_cursor_vline.set_alpha(cursor_alpha)
            self._live_cursor_vline.set_linewidth(cursor_width)
        else:
            self._live_cursor_vline = self.ax.axvline(
                event.xdata, color=cursor_color,
                linestyle='-', linewidth=cursor_width, alpha=cursor_alpha
            )

        self.canvas.draw_idle()

    def _find_nearest_wf(
        self, x: float, y: float
    ) -> Optional[Tuple[str, float, str]]:
        """Find the nearest visible line to the mouse position.

        Checks envelope lines when visible, individual waveforms otherwise.
        Returns (name, y_value, color_hex) or None if nothing is close.
        """
        y_min, y_max = self.ax.get_ylim()
        threshold = (y_max - y_min) * CURSOR_PROXIMITY_THRESHOLD

        best_dist = threshold
        best_result = None

        # Use cached waveform data from last plot update
        wf_data = self._cached_wf_data
        if not wf_data:
            return None

        # Check envelope lines when they're visible
        if app_state.can_show_envelopes() and wf_data:
            env_candidates: list[Tuple[str, float, str]] = []

            if app_state.show_max_env:
                _, max_env = compute_max_env(wf_data)
                env_y = float(np.interp(x, wf_data[0][0], max_env))
                env_candidates.append(("Max Envelope", env_y, COLOR_SUCCESS))

            if app_state.show_min_env:
                _, min_env = compute_min_env(wf_data)
                env_y = float(np.interp(x, wf_data[0][0], min_env))
                env_candidates.append(("Min Envelope", env_y, COLOR_ERROR))

            if app_state.show_rms_env:
                _, rms_env = compute_rms_env(wf_data)
                env_y = float(np.interp(x, wf_data[0][0], rms_env))
                env_candidates.append(("RMS Envelope", env_y, COLOR_RMS))

            for name, env_y, color in env_candidates:
                dist = abs(y - env_y)
                if dist < best_dist:
                    best_dist = dist
                    best_result = (name, env_y, color)

        # Check individual waveforms when they're visible
        if not app_state.hide_src_wfs:
            for wf, (time, amp) in zip(
                [w for w in app_state.wfs if w.enabled], wf_data
            ):
                wf_y = float(np.interp(x, time, amp))
                dist = abs(y - wf_y)
                if dist < best_dist:
                    best_dist = dist
                    color_hex = '#{:02x}{:02x}{:02x}'.format(*wf.color)
                    best_result = (wf.display_name, wf_y, color_hex)

        return best_result

    def _remove_highlight_marker(self):
        """Remove the highlight dot from the plot."""
        if self._highlight_marker is not None:
            try:
                self._highlight_marker.remove()
            except ValueError:
                pass
            self._highlight_marker = None

    def _on_plot_click(self, event: Any):
        """Handle click on the plot to pin a reference cursor."""
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return

        # Remove old pinned cursor line
        if self._pinned_cursor_vline and self._pinned_cursor_vline in self.ax.lines:
            self._pinned_cursor_vline.remove()

        # Pin a reference cursor at click position
        self._pinned_cursor_x = event.xdata
        self._pinned_cursor_vline = self.ax.axvline(
            event.xdata, color=COLOR_CURSOR_PINNED,
            linestyle='--', linewidth=1, alpha=0.7
        )

        self.canvas.draw_idle()

    def _redraw_cursors(self):
        """Re-draw cursor lines after ax.clear() and update readout."""
        # ax.clear() already removed these, reset references
        self._highlight_marker = None

        # Redraw pinned cursor
        if self._pinned_cursor_x is not None:
            self._pinned_cursor_vline = self.ax.axvline(
                self._pinned_cursor_x, color=COLOR_CURSOR_PINNED,
                linestyle='--', linewidth=1, alpha=0.7
            )
        # Redraw live cursor (highlight recalculated on next mouse move)
        if self._live_cursor_x is not None:
            self._live_cursor_vline = self.ax.axvline(
                self._live_cursor_x, color=COLOR_CURSOR_DEFAULT,
                linestyle='-', linewidth=1, alpha=0.5
            )

    def _update_status_bar(self):
        """Update status bar with current info."""
        num_wfs = len(app_state.wfs)
        self.status_bar.configure(text=f"Waveforms: {num_wfs}/{app_state.MAX_WFS}")
