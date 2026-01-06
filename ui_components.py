"""
UI components for the Real-Time Waveform Visualizer.

This module contains all CustomTkinter UI creation and callback logic.
"""

import customtkinter as ctk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from app_state import (
    app_state, DEFAULT_TIME_SPAN,
    TIME_SPAN_MIN, TIME_SPAN_MAX, TIME_SPAN_STEP,
    FREQUENCY_MIN, FREQUENCY_MAX, FREQUENCY_STEP,
    AMPLITUDE_MIN, AMPLITUDE_MAX, AMPLITUDE_STEP,
    DUTY_CYCLE_MIN, DUTY_CYCLE_MAX, DUTY_CYCLE_STEP
)
from waveform_generator import generate_waveform, compute_max_envelope, compute_min_envelope
from data_export import export_to_csv, prepare_waveform_for_export


# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Color constants
SECTION_HEADER_COLOR = "#FFFF00"  # Yellow
ENABLED_TEXT_COLOR = "#FFFFFF"
DISABLED_TEXT_COLOR = "#808080"


class WaveformApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Waveform Generator/Analyzer")
        self.geometry("1200x900")
        self.minsize(1000, 700)

        # Configure grid weights
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Store widget references
        self.waveform_buttons = []
        self.toggle_buttons = []
        self.remove_buttons = []

        # Create UI components
        self._create_sidebar()
        self._create_plot_area()
        self._create_status_bar()

        # Initialize UI state
        self._update_waveform_list()
        self._update_waveform_parameters()
        self._update_envelope_controls()
        self._update_add_button()
        self._update_all_plots()

    def _create_sidebar(self):
        """Create the sidebar with all controls."""
        # Sidebar frame with scrolling
        self.sidebar = ctk.CTkScrollableFrame(self, width=330)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 0))

        # === Display Controls ===
        self._add_section_header("Display Controls")

        # Time Span
        ctk.CTkLabel(self.sidebar, text="Time Span (s)").pack(anchor="w", pady=(5, 2))
        time_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        time_frame.pack(fill="x", pady=(0, 5))

        self.time_span_entry = ctk.CTkEntry(time_frame, width=120)
        self.time_span_entry.pack(side="left", padx=(0, 5))
        self.time_span_entry.insert(0, str(DEFAULT_TIME_SPAN))
        self.time_span_entry.bind("<Return>", self._on_time_span_enter)
        self.time_span_entry.bind("<FocusOut>", self._on_time_span_enter)

        self.time_dec_btn = ctk.CTkButton(
            time_frame, text="-", width=30,
            command=self._on_time_span_decrement
        )
        self.time_dec_btn.pack(side="left", padx=2)

        self.time_inc_btn = ctk.CTkButton(
            time_frame, text="+", width=30,
            command=self._on_time_span_increment
        )
        self.time_inc_btn.pack(side="left", padx=2)

        # Show Grid checkbox
        self.show_grid_var = ctk.BooleanVar(value=True)
        self.show_grid_cb = ctk.CTkCheckBox(
            self.sidebar, text="Show Grid",
            variable=self.show_grid_var,
            command=self._on_show_grid_changed
        )
        self.show_grid_cb.pack(anchor="w", pady=2)

        # Reset View button
        ctk.CTkButton(
            self.sidebar, text="Reset View",
            command=self._on_reset_view
        ).pack(fill="x", pady=(5, 10))

        # === Advanced ===
        self._add_section_header("Advanced")

        # Max Envelope checkbox
        env_frame1 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        env_frame1.pack(fill="x", pady=2)
        self.max_env_var = ctk.BooleanVar(value=False)
        self.max_env_cb = ctk.CTkCheckBox(
            env_frame1, text="",
            variable=self.max_env_var,
            command=self._on_max_envelope_changed,
            width=24
        )
        self.max_env_cb.pack(side="left")
        self.max_env_label = ctk.CTkLabel(env_frame1, text="Show Max Envelope")
        self.max_env_label.pack(side="left")

        # Min Envelope checkbox
        env_frame2 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        env_frame2.pack(fill="x", pady=2)
        self.min_env_var = ctk.BooleanVar(value=False)
        self.min_env_cb = ctk.CTkCheckBox(
            env_frame2, text="",
            variable=self.min_env_var,
            command=self._on_min_envelope_changed,
            width=24
        )
        self.min_env_cb.pack(side="left")
        self.min_env_label = ctk.CTkLabel(env_frame2, text="Show Min Envelope")
        self.min_env_label.pack(side="left")

        # Hide Source Waveforms checkbox
        env_frame3 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        env_frame3.pack(fill="x", pady=(2, 10))
        self.hide_source_var = ctk.BooleanVar(value=False)
        self.hide_source_cb = ctk.CTkCheckBox(
            env_frame3, text="",
            variable=self.hide_source_var,
            command=self._on_hide_source_changed,
            width=24
        )
        self.hide_source_cb.pack(side="left")
        self.hide_source_label = ctk.CTkLabel(env_frame3, text="Hide Source Waveforms")
        self.hide_source_label.pack(side="left")

        # === Waveforms ===
        self._add_section_header("Waveforms")

        # Add Waveform button
        self.add_wave_btn = ctk.CTkButton(
            self.sidebar, text="+ Add Waveform",
            command=self._on_add_waveform
        )
        self.add_wave_btn.pack(fill="x", pady=(5, 5))

        # Waveform list container
        self.waveform_list_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.waveform_list_frame.pack(fill="x", pady=(0, 10))

        # === Waveform Parameters ===
        self._add_section_header("Waveform Parameters")

        # Wave Type
        ctk.CTkLabel(self.sidebar, text="Type").pack(anchor="w", pady=(5, 2))
        self.wave_type_combo = ctk.CTkComboBox(
            self.sidebar,
            values=["Sine", "Square", "Sawtooth", "Triangle"],
            command=self._on_wave_type_changed,
            width=200
        )
        self.wave_type_combo.pack(anchor="w", pady=(0, 5))
        self.wave_type_combo.set("Sine")

        # Frequency
        ctk.CTkLabel(self.sidebar, text="Frequency (Hz)").pack(anchor="w", pady=(5, 2))
        freq_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        freq_frame.pack(fill="x", pady=(0, 5))

        self.freq_entry = ctk.CTkEntry(freq_frame, width=120)
        self.freq_entry.pack(side="left", padx=(0, 5))
        self.freq_entry.insert(0, str(FREQUENCY_MIN))
        self.freq_entry.bind("<Return>", self._on_frequency_enter)
        self.freq_entry.bind("<FocusOut>", self._on_frequency_enter)

        self.freq_dec_btn = ctk.CTkButton(
            freq_frame, text="-", width=30,
            command=self._on_frequency_decrement
        )
        self.freq_dec_btn.pack(side="left", padx=2)

        self.freq_inc_btn = ctk.CTkButton(
            freq_frame, text="+", width=30,
            command=self._on_frequency_increment
        )
        self.freq_inc_btn.pack(side="left", padx=2)

        # Amplitude
        ctk.CTkLabel(self.sidebar, text="Amplitude").pack(anchor="w", pady=(5, 2))
        amp_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        amp_frame.pack(fill="x", pady=(0, 5))

        self.amp_entry = ctk.CTkEntry(amp_frame, width=120)
        self.amp_entry.pack(side="left", padx=(0, 5))
        self.amp_entry.insert(0, "5.0")
        self.amp_entry.bind("<Return>", self._on_amplitude_enter)
        self.amp_entry.bind("<FocusOut>", self._on_amplitude_enter)

        self.amp_dec_btn = ctk.CTkButton(
            amp_frame, text="-", width=30,
            command=self._on_amplitude_decrement
        )
        self.amp_dec_btn.pack(side="left", padx=2)

        self.amp_inc_btn = ctk.CTkButton(
            amp_frame, text="+", width=30,
            command=self._on_amplitude_increment
        )
        self.amp_inc_btn.pack(side="left", padx=2)

        # Duty Cycle (hidden by default, shown for Square waves)
        self.duty_label = ctk.CTkLabel(self.sidebar, text="Duty Cycle (%)")
        self.duty_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")

        self.duty_entry = ctk.CTkEntry(self.duty_frame, width=120)
        self.duty_entry.pack(side="left", padx=(0, 5))
        self.duty_entry.insert(0, "50.0")
        self.duty_entry.bind("<Return>", self._on_duty_cycle_enter)
        self.duty_entry.bind("<FocusOut>", self._on_duty_cycle_enter)

        self.duty_dec_btn = ctk.CTkButton(
            self.duty_frame, text="-", width=30,
            command=self._on_duty_cycle_decrement
        )
        self.duty_dec_btn.pack(side="left", padx=2)

        self.duty_inc_btn = ctk.CTkButton(
            self.duty_frame, text="+", width=30,
            command=self._on_duty_cycle_increment
        )
        self.duty_inc_btn.pack(side="left", padx=2)

        # === Export ===
        self._add_section_header("Export")

        ctk.CTkButton(
            self.sidebar, text="Export to CSV",
            command=self._on_export_clicked
        ).pack(fill="x", pady=(5, 5))

        self.export_status = ctk.CTkLabel(
            self.sidebar, text="Status: Ready",
            text_color="#00FF00"
        )
        self.export_status.pack(anchor="w", pady=(5, 10))

    def _create_plot_area(self):
        """Create the matplotlib plot area."""
        # Plot container
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 0))
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

        # Create matplotlib figure with dark theme
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(8, 6), facecolor='#1a1a1a')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a1a1a')

        # Configure axes
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3, color='#666666')

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = ctk.CTkLabel(
            self, text="Waveforms: 1/5",
            anchor="w"
        )
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))

    def _add_section_header(self, text: str):
        """Add a section header with separator."""
        # Separator line
        separator = ctk.CTkFrame(self.sidebar, height=1, fg_color="#666666")
        separator.pack(fill="x", pady=(10, 5))

        # Header text
        header = ctk.CTkLabel(
            self.sidebar, text=text,
            text_color=SECTION_HEADER_COLOR,
            font=ctk.CTkFont(weight="bold")
        )
        header.pack(anchor="w")

    # === Callback Methods ===

    def _on_time_span_enter(self, event=None):
        """Handle time span entry."""
        try:
            value = float(self.time_span_entry.get())
            value = max(TIME_SPAN_MIN, min(TIME_SPAN_MAX, value))
            app_state.set_time_span(value)
            self.time_span_entry.delete(0, "end")
            self.time_span_entry.insert(0, f"{value:.1f}")
            self._update_time_span_buttons()
            self._update_all_plots()
        except ValueError:
            self.time_span_entry.delete(0, "end")
            self.time_span_entry.insert(0, f"{app_state.time_span:.1f}")

    def _on_time_span_increment(self):
        """Increment time span."""
        new_value = min(TIME_SPAN_MAX, app_state.time_span + TIME_SPAN_STEP)
        app_state.set_time_span(new_value)
        self.time_span_entry.delete(0, "end")
        self.time_span_entry.insert(0, f"{new_value:.1f}")
        self._update_time_span_buttons()
        self._update_all_plots()

    def _on_time_span_decrement(self):
        """Decrement time span."""
        new_value = max(TIME_SPAN_MIN, app_state.time_span - TIME_SPAN_STEP)
        app_state.set_time_span(new_value)
        self.time_span_entry.delete(0, "end")
        self.time_span_entry.insert(0, f"{new_value:.1f}")
        self._update_time_span_buttons()
        self._update_all_plots()

    def _on_show_grid_changed(self):
        """Handle grid visibility toggle."""
        app_state.show_grid = self.show_grid_var.get()
        self._update_all_plots()

    def _on_reset_view(self):
        """Reset view to defaults."""
        app_state.set_time_span(DEFAULT_TIME_SPAN)
        self.time_span_entry.delete(0, "end")
        self.time_span_entry.insert(0, f"{DEFAULT_TIME_SPAN:.1f}")
        self._update_time_span_buttons()
        self._update_all_plots()

    def _on_max_envelope_changed(self):
        """Handle max envelope toggle."""
        app_state.show_max_envelope = self.max_env_var.get()
        self._update_envelope_controls()
        self._update_all_plots()

    def _on_min_envelope_changed(self):
        """Handle min envelope toggle."""
        app_state.show_min_envelope = self.min_env_var.get()
        self._update_envelope_controls()
        self._update_all_plots()

    def _on_hide_source_changed(self):
        """Handle hide source waveforms toggle."""
        app_state.hide_source_waveforms = self.hide_source_var.get()
        self._update_waveform_management_controls()
        self._update_all_plots()

    def _on_add_waveform(self):
        """Add a new waveform."""
        new_waveform = app_state.add_waveform()
        if new_waveform:
            self._update_waveform_list()
            self._update_waveform_parameters()
            self._update_envelope_controls()
            self._update_all_plots()
            self._update_add_button()

    def _on_remove_waveform(self, waveform_id: int):
        """Remove a waveform."""
        if app_state.remove_waveform(waveform_id):
            self._update_waveform_list()
            self._update_waveform_parameters()
            self._update_envelope_controls()
            self._update_all_plots()
            self._update_add_button()

    def _on_toggle_waveform(self, waveform_id: int):
        """Toggle waveform visibility."""
        waveform = app_state.get_waveform(waveform_id)
        if waveform:
            waveform.enabled = not waveform.enabled
            self._update_envelope_controls()
            self._update_all_plots()
            self._update_waveform_list()

    def _on_select_waveform(self, waveform_id: int):
        """Select a waveform for editing."""
        app_state.active_waveform_index = waveform_id
        self._update_waveform_parameters()
        self._update_waveform_list()

    def _on_wave_type_changed(self, value: str):
        """Handle wave type change."""
        waveform = app_state.get_active_waveform()
        if waveform:
            waveform.wave_type = value.lower()
            self._update_waveform_parameters()
            self._update_all_plots()
            self._update_waveform_list()

    def _on_frequency_enter(self, event=None):
        """Handle frequency entry."""
        waveform = app_state.get_active_waveform()
        if waveform:
            try:
                value = float(self.freq_entry.get())
                value = max(FREQUENCY_MIN, min(FREQUENCY_MAX, value))
                waveform.frequency = value
                self.freq_entry.delete(0, "end")
                self.freq_entry.insert(0, f"{value:.1f}")
                self._update_waveform_parameters()
                self._update_all_plots()
                self._update_waveform_list()
            except ValueError:
                self.freq_entry.delete(0, "end")
                self.freq_entry.insert(0, f"{waveform.frequency:.1f}")

    def _on_frequency_increment(self):
        """Increment frequency."""
        waveform = app_state.get_active_waveform()
        if waveform:
            new_value = min(FREQUENCY_MAX, waveform.frequency + FREQUENCY_STEP)
            waveform.frequency = new_value
            self.freq_entry.delete(0, "end")
            self.freq_entry.insert(0, f"{new_value:.1f}")
            self._update_waveform_parameters()
            self._update_all_plots()
            self._update_waveform_list()

    def _on_frequency_decrement(self):
        """Decrement frequency."""
        waveform = app_state.get_active_waveform()
        if waveform:
            new_value = max(FREQUENCY_MIN, waveform.frequency - FREQUENCY_STEP)
            waveform.frequency = new_value
            self.freq_entry.delete(0, "end")
            self.freq_entry.insert(0, f"{new_value:.1f}")
            self._update_waveform_parameters()
            self._update_all_plots()
            self._update_waveform_list()

    def _on_amplitude_enter(self, event=None):
        """Handle amplitude entry."""
        waveform = app_state.get_active_waveform()
        if waveform:
            try:
                value = float(self.amp_entry.get())
                value = max(AMPLITUDE_MIN, min(AMPLITUDE_MAX, value))
                waveform.amplitude = value
                self.amp_entry.delete(0, "end")
                self.amp_entry.insert(0, f"{value:.1f}")
                self._update_waveform_parameters()
                self._update_all_plots()
            except ValueError:
                self.amp_entry.delete(0, "end")
                self.amp_entry.insert(0, f"{waveform.amplitude:.1f}")

    def _on_amplitude_increment(self):
        """Increment amplitude."""
        waveform = app_state.get_active_waveform()
        if waveform:
            new_value = min(AMPLITUDE_MAX, waveform.amplitude + AMPLITUDE_STEP)
            waveform.amplitude = new_value
            self.amp_entry.delete(0, "end")
            self.amp_entry.insert(0, f"{new_value:.1f}")
            self._update_waveform_parameters()
            self._update_all_plots()

    def _on_amplitude_decrement(self):
        """Decrement amplitude."""
        waveform = app_state.get_active_waveform()
        if waveform:
            new_value = max(AMPLITUDE_MIN, waveform.amplitude - AMPLITUDE_STEP)
            waveform.amplitude = new_value
            self.amp_entry.delete(0, "end")
            self.amp_entry.insert(0, f"{new_value:.1f}")
            self._update_waveform_parameters()
            self._update_all_plots()

    def _on_duty_cycle_enter(self, event=None):
        """Handle duty cycle entry."""
        waveform = app_state.get_active_waveform()
        if waveform:
            try:
                value = float(self.duty_entry.get())
                value = max(DUTY_CYCLE_MIN, min(DUTY_CYCLE_MAX, value))
                waveform.duty_cycle = value
                self.duty_entry.delete(0, "end")
                self.duty_entry.insert(0, f"{value:.1f}")
                self._update_waveform_parameters()
                self._update_all_plots()
            except ValueError:
                self.duty_entry.delete(0, "end")
                self.duty_entry.insert(0, f"{waveform.duty_cycle:.1f}")

    def _on_duty_cycle_increment(self):
        """Increment duty cycle."""
        waveform = app_state.get_active_waveform()
        if waveform:
            new_value = min(DUTY_CYCLE_MAX, waveform.duty_cycle + DUTY_CYCLE_STEP)
            waveform.duty_cycle = new_value
            self.duty_entry.delete(0, "end")
            self.duty_entry.insert(0, f"{new_value:.1f}")
            self._update_waveform_parameters()
            self._update_all_plots()

    def _on_duty_cycle_decrement(self):
        """Decrement duty cycle."""
        waveform = app_state.get_active_waveform()
        if waveform:
            new_value = max(DUTY_CYCLE_MIN, waveform.duty_cycle - DUTY_CYCLE_STEP)
            waveform.duty_cycle = new_value
            self.duty_entry.delete(0, "end")
            self.duty_entry.insert(0, f"{new_value:.1f}")
            self._update_waveform_parameters()
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
        waveforms_to_export = []
        waveform_arrays = []
        for waveform in app_state.get_enabled_waveforms():
            time, amplitude = generate_waveform(
                waveform.wave_type,
                waveform.frequency,
                waveform.amplitude,
                waveform.duty_cycle,
                app_state.time_span,
                app_state.sample_rate
            )
            waveform_arrays.append((time, amplitude))

            name = f"Waveform_{waveform.id + 1}_{waveform.wave_type.capitalize()}"
            export_data = prepare_waveform_for_export(
                name, time, amplitude,
                waveform.wave_type,
                waveform.frequency,
                waveform.amplitude,
                waveform.duty_cycle
            )
            waveforms_to_export.append(export_data)

        # Collect envelope data if enabled
        envelopes_to_export = []
        if app_state.can_show_envelopes() and waveform_arrays:
            if app_state.show_max_envelope:
                time, max_env = compute_max_envelope(waveform_arrays)
                envelopes_to_export.append(("Max_Envelope", time, max_env))

            if app_state.show_min_envelope:
                time, min_env = compute_min_envelope(waveform_arrays)
                envelopes_to_export.append(("Min_Envelope", time, min_env))

        # Export
        success, message = export_to_csv(
            filename,
            waveforms_to_export,
            envelopes_to_export if envelopes_to_export else None,
            app_state.sample_rate,
            app_state.time_span
        )

        # Update status
        if success:
            self.export_status.configure(text=message, text_color="#00FF00")
        else:
            self.export_status.configure(text=message, text_color="#FF0000")

    # === UI Update Methods ===

    def _update_all_plots(self):
        """Regenerate and update all waveform plots."""
        self.ax.clear()

        # Configure axes
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlim(0, app_state.time_span)
        self.ax.set_ylim(-12, 12)
        if app_state.show_grid:
            self.ax.grid(visible=True, alpha=0.3, color='#666666')
        else:
            self.ax.grid(visible=False)

        # Generate and plot enabled waveforms
        waveform_data = []
        for waveform in app_state.waveforms:
            if waveform.enabled:
                time, amplitude = generate_waveform(
                    waveform.wave_type,
                    waveform.frequency,
                    waveform.amplitude,
                    waveform.duty_cycle,
                    app_state.time_span,
                    app_state.sample_rate
                )
                waveform_data.append((time, amplitude))

                # Only plot if not hiding source waveforms
                if not app_state.hide_source_waveforms:
                    # Convert RGB tuple to matplotlib color format
                    color = tuple(c / 255 for c in waveform.color)
                    label = f"Waveform {waveform.id + 1}"
                    self.ax.plot(time, amplitude, color=color, label=label, linewidth=2)

        # Plot envelopes
        if app_state.can_show_envelopes() and waveform_data:
            if app_state.show_max_envelope:
                time, max_env = compute_max_envelope(waveform_data)
                self.ax.plot(time, max_env, color='darkblue', label='Max Envelope',
                           linewidth=2, linestyle='--')

            if app_state.show_min_envelope:
                time, min_env = compute_min_envelope(waveform_data)
                self.ax.plot(time, min_env, color='red', label='Min Envelope',
                           linewidth=2, linestyle='--')

        # Add legend if there are any lines
        if self.ax.get_lines():
            self.ax.legend(loc='upper right')

        # Redraw canvas
        self.canvas.draw()

        # Update status bar
        self._update_status_bar()

    def _update_waveform_list(self):
        """Update the waveform list UI."""
        # Clear existing widgets safely
        children = list(self.waveform_list_frame.winfo_children())
        for widget in children:
            try:
                widget.destroy()
            except Exception:
                pass

        self.waveform_buttons = []
        self.toggle_buttons = []
        self.remove_buttons = []

        for waveform in app_state.waveforms:
            row_frame = ctk.CTkFrame(self.waveform_list_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)

            # Selection button
            is_selected = waveform.id == app_state.active_waveform_index
            fg_color = "#3d3d6b" if is_selected else "#2b2b2b"
            border_color = "#6496ff" if is_selected else "#2b2b2b"
            border_width = 2 if is_selected else 0

            wave_btn = ctk.CTkButton(
                row_frame,
                text=f"Waveform {waveform.id + 1}",
                width=180,
                fg_color=fg_color,
                border_color=border_color,
                border_width=border_width,
                command=lambda wid=waveform.id: self._on_select_waveform(wid)
            )
            wave_btn.pack(side="left", padx=(0, 5))
            self.waveform_buttons.append(wave_btn)

            # Toggle button
            toggle_text = "ON" if waveform.enabled else "OFF"
            toggle_color = "#009600" if waveform.enabled else "#646464"
            toggle_btn = ctk.CTkButton(
                row_frame,
                text=toggle_text,
                width=40,
                fg_color=toggle_color,
                command=lambda wid=waveform.id: self._on_toggle_waveform(wid)
            )
            toggle_btn.pack(side="left", padx=2)
            self.toggle_buttons.append(toggle_btn)

            # Remove button (only show if more than 1 waveform)
            if len(app_state.waveforms) > app_state.MIN_WAVEFORMS:
                is_enabled = not app_state.hide_source_waveforms
                remove_btn = ctk.CTkButton(
                    row_frame,
                    text="X",
                    width=30,
                    fg_color="#8B0000" if is_enabled else "#646464",
                    state="normal" if is_enabled else "disabled",
                    command=lambda wid=waveform.id: self._on_remove_waveform(wid)
                )
                remove_btn.pack(side="left", padx=2)
                self.remove_buttons.append(remove_btn)

    def _update_waveform_parameters(self):
        """Update waveform parameter inputs based on active waveform."""
        waveform = app_state.get_active_waveform()
        if not waveform:
            return

        # Update entry fields
        self.freq_entry.delete(0, "end")
        self.freq_entry.insert(0, f"{waveform.frequency:.1f}")

        self.amp_entry.delete(0, "end")
        self.amp_entry.insert(0, f"{waveform.amplitude:.1f}")

        self.duty_entry.delete(0, "end")
        self.duty_entry.insert(0, f"{waveform.duty_cycle:.1f}")

        self.wave_type_combo.set(waveform.wave_type.capitalize())

        # Update button states
        self._update_frequency_buttons()
        self._update_amplitude_buttons()
        self._update_duty_cycle_buttons()

        # Show/hide duty cycle for square waves
        needs_duty = waveform.wave_type.lower() == 'square'
        if needs_duty:
            self.duty_label.pack(anchor="w", pady=(5, 2))
            self.duty_frame.pack(fill="x", pady=(0, 5))
        else:
            self.duty_label.pack_forget()
            self.duty_frame.pack_forget()

    def _update_time_span_buttons(self):
        """Update time span button states."""
        at_min = app_state.time_span <= TIME_SPAN_MIN
        at_max = app_state.time_span >= TIME_SPAN_MAX
        self.time_dec_btn.configure(state="disabled" if at_min else "normal")
        self.time_inc_btn.configure(state="disabled" if at_max else "normal")

    def _update_frequency_buttons(self):
        """Update frequency button states."""
        waveform = app_state.get_active_waveform()
        if waveform:
            at_min = waveform.frequency <= FREQUENCY_MIN
            at_max = waveform.frequency >= FREQUENCY_MAX
            self.freq_dec_btn.configure(state="disabled" if at_min else "normal")
            self.freq_inc_btn.configure(state="disabled" if at_max else "normal")

    def _update_amplitude_buttons(self):
        """Update amplitude button states."""
        waveform = app_state.get_active_waveform()
        if waveform:
            at_min = waveform.amplitude <= AMPLITUDE_MIN
            at_max = waveform.amplitude >= AMPLITUDE_MAX
            self.amp_dec_btn.configure(state="disabled" if at_min else "normal")
            self.amp_inc_btn.configure(state="disabled" if at_max else "normal")

    def _update_duty_cycle_buttons(self):
        """Update duty cycle button states."""
        waveform = app_state.get_active_waveform()
        if waveform:
            at_min = waveform.duty_cycle <= DUTY_CYCLE_MIN
            at_max = waveform.duty_cycle >= DUTY_CYCLE_MAX
            self.duty_dec_btn.configure(state="disabled" if at_min else "normal")
            self.duty_inc_btn.configure(state="disabled" if at_max else "normal")

    def _update_envelope_controls(self):
        """Enable/disable envelope checkboxes based on number of enabled waveforms."""
        can_show = app_state.can_show_envelopes()

        # Update max envelope checkbox
        self.max_env_cb.configure(state="normal" if can_show else "disabled")
        self.max_env_label.configure(
            text_color=ENABLED_TEXT_COLOR if can_show else DISABLED_TEXT_COLOR
        )

        # Update min envelope checkbox
        self.min_env_cb.configure(state="normal" if can_show else "disabled")
        self.min_env_label.configure(
            text_color=ENABLED_TEXT_COLOR if can_show else DISABLED_TEXT_COLOR
        )

        # Hide source checkbox requires at least one envelope to be shown
        can_hide_source = can_show and (app_state.show_max_envelope or app_state.show_min_envelope)
        self.hide_source_cb.configure(state="normal" if can_hide_source else "disabled")
        self.hide_source_label.configure(
            text_color=ENABLED_TEXT_COLOR if can_hide_source else DISABLED_TEXT_COLOR
        )

        # If hide source becomes disabled, turn it off
        if not can_hide_source:
            if app_state.hide_source_waveforms:
                app_state.hide_source_waveforms = False
                self._update_waveform_management_controls()
            self.hide_source_var.set(False)

        if not can_show:
            app_state.show_max_envelope = False
            app_state.show_min_envelope = False
            self.max_env_var.set(False)
            self.min_env_var.set(False)

    def _update_add_button(self):
        """Enable/disable add waveform button based on max limit."""
        can_add = len(app_state.waveforms) < app_state.MAX_WAVEFORMS
        self.add_wave_btn.configure(state="normal" if can_add else "disabled")

    def _update_waveform_management_controls(self):
        """Enable/disable waveform management controls based on hide_source state."""
        enabled = not app_state.hide_source_waveforms
        self.add_wave_btn.configure(state="normal" if enabled else "disabled")
        self._update_waveform_list()

    def _update_status_bar(self):
        """Update status bar with current info."""
        num_waveforms = len(app_state.waveforms)
        self.status_bar.configure(text=f"Waveforms: {num_waveforms}/{app_state.MAX_WAVEFORMS}")
