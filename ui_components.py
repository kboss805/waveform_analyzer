"""
UI components and callbacks for the Real-Time Waveform Visualizer.

This module contains all DearPyGui UI creation and callback logic.
"""

import dearpygui.dearpygui as dpg
from typing import Optional
from app_state import app_state, WaveformState, DEFAULT_TIME_SPAN
from waveform_generator import generate_waveform, compute_max_envelope, compute_min_envelope
from data_export import export_to_csv, prepare_waveform_for_export


# UI element tags
MAIN_WINDOW = "main_window"
PLOT_TAG = "main_plot"
Y_AXIS_TAG = "y_axis"
STATUS_BAR = "status_bar"
TIME_SPAN_SLIDER = "time_span_slider"
TIME_SPAN_DEC_BTN = "time_span_dec_btn"
TIME_SPAN_INPUT = "time_span_input"
TIME_SPAN_INC_BTN = "time_span_inc_btn"
TIME_SPAN_GROUP = "time_span_group"
FREQ_SLIDER = "freq_slider"
AMP_SLIDER = "amp_slider"
DUTY_SLIDER = "duty_slider"
FREQ_DEC_BTN = "freq_dec_btn"
FREQ_INPUT = "freq_input"
FREQ_INC_BTN = "freq_inc_btn"
AMP_DEC_BTN = "amp_dec_btn"
AMP_INPUT = "amp_input"
AMP_INC_BTN = "amp_inc_btn"
DUTY_DEC_BTN = "duty_dec_btn"
DUTY_INPUT = "duty_input"
DUTY_INC_BTN = "duty_inc_btn"
FREQ_GROUP = "freq_group"
AMP_GROUP = "amp_group"
DUTY_GROUP = "duty_group"
DUTY_LABEL = "duty_label"
WAVE_TYPE_COMBO = "wave_type_combo"
ADD_WAVE_BTN = "add_wave_btn"
EXPORT_STATUS = "export_status"
EXPORT_BTN = "export_btn"
FILE_DIALOG = "file_dialog"
WAVEFORM_LIST_GROUP = "waveform_list_group"
SHOW_GRID_CHECKBOX = "show_grid_checkbox"
MAX_ENV_CHECKBOX = "max_env_checkbox"
MIN_ENV_CHECKBOX = "min_env_checkbox"
HIDE_SOURCE_CHECKBOX = "hide_source_checkbox"
MAX_ENV_LABEL = "max_env_label"
MIN_ENV_LABEL = "min_env_label"
HIDE_SOURCE_LABEL = "hide_source_label"
SIDEBAR_CHILD = "sidebar_child"
SPLITTER = "splitter"

# Sidebar width state
sidebar_width = 350
is_dragging_splitter = False

# Toggle button themes
toggle_on_theme = None
toggle_off_theme = None

# Disabled checkbox theme
disabled_checkbox_theme = None

# Disabled button theme
disabled_button_theme = None

# Waveform button themes
waveform_selected_theme = None
waveform_unselected_theme = None

# Default view settings (Y-axis limits)
DEFAULT_Y_MIN = -12
DEFAULT_Y_MAX = 12

# Label colors
ENABLED_LABEL_COLOR = (255, 255, 255, 255)  # White
DISABLED_LABEL_COLOR = (128, 128, 128, 255)  # Light gray


def update_all_plots() -> None:
    """Regenerate and update all waveform plots and envelopes."""
    # Clear existing series
    if dpg.does_item_exist(Y_AXIS_TAG):
        children = dpg.get_item_children(Y_AXIS_TAG, slot=1)
        if children:
            for child in children:
                dpg.delete_item(child)

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

            # Only add line series if not hiding source waveforms
            if not app_state.hide_source_waveforms:
                # Create series
                series_tag = f"wave_{waveform.id}_series"
                label = f"Waveform {waveform.id + 1}"

                # Convert color to normalized values (0-1)
                color_normalized = tuple(c / 255.0 for c in waveform.color) + (1.0,)
                color_int = tuple(int(c) for c in waveform.color) + (255,)

                dpg.add_line_series(
                    time.tolist(),
                    amplitude.tolist(),
                    label=label,
                    parent=Y_AXIS_TAG,
                    tag=series_tag
                )
                dpg.bind_item_theme(series_tag, create_line_theme(color_int))

    # Update envelopes
    if app_state.can_show_envelopes():
        if app_state.show_max_envelope and waveform_data:
            time, max_env = compute_max_envelope(waveform_data)
            dpg.add_line_series(
                time.tolist(),
                max_env.tolist(),
                label="Max Envelope",
                parent=Y_AXIS_TAG,
                tag="max_env_series"
            )
            dpg.bind_item_theme("max_env_series", create_line_theme((0, 0, 139, 255), thickness=2))

        if app_state.show_min_envelope and waveform_data:
            time, min_env = compute_min_envelope(waveform_data)
            dpg.add_line_series(
                time.tolist(),
                min_env.tolist(),
                label="Min Envelope",
                parent=Y_AXIS_TAG,
                tag="min_env_series"
            )
            dpg.bind_item_theme("min_env_series", create_line_theme((255, 0, 0, 255), thickness=2))

    # Set fixed Y-axis limits
    dpg.set_axis_limits(Y_AXIS_TAG, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    # Set X-axis limits based on time span
    x_axis = dpg.get_item_children(PLOT_TAG, slot=1)[0]
    dpg.set_axis_limits(x_axis, 0, app_state.time_span)

    update_status_bar()


def create_line_theme(color: tuple, thickness: float = 2.0) -> str:
    """
    Create a theme for line series.

    Args:
        color: RGBA color tuple (0-255)
        thickness: Line thickness

    Returns:
        Theme tag
    """
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, thickness, category=dpg.mvThemeCat_Plots)
    return theme


def create_toggle_button_themes():
    """Create themes for ON/OFF toggle buttons."""
    # ON button theme (green)
    with dpg.theme() as on_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 150, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 180, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 120, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))

    # OFF button theme (gray)
    with dpg.theme() as off_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (100, 100, 100, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (120, 120, 120, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (80, 80, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (200, 200, 200, 255))

    return on_theme, off_theme


def create_waveform_button_themes():
    """Create themes for selected and unselected waveform buttons."""
    # Selected waveform theme (highlighted with border)
    with dpg.theme() as selected_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 100, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 70, 120, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 50, 90, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (100, 150, 255, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2.0)

    # Unselected waveform theme (default appearance)
    with dpg.theme() as unselected_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0.0)

    return selected_theme, unselected_theme


def create_disabled_checkbox_theme():
    """Create theme for disabled checkboxes to prevent hover effects."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvCheckbox, enabled_state=False):
            # Keep the same color for disabled state and hover to prevent highlighting
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (51, 51, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (51, 51, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (128, 128, 128, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (128, 128, 128, 255))  # Light grey text
    return theme


def create_disabled_button_theme():
    """Create theme for disabled buttons to prevent hover effects and show grey text."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            # Keep the same color for disabled state and hover to prevent highlighting
            dpg.add_theme_color(dpg.mvThemeCol_Button, (51, 51, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 51, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (51, 51, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (128, 128, 128, 255))  # Light grey text
    return theme


def on_time_span_changed(sender, value):
    """Callback for time span slider."""
    app_state.set_time_span(value)
    update_all_plots()


def on_time_span_increment():
    """Increment time span by step amount."""
    new_value = min(120.0, app_state.time_span + 0.5)
    app_state.set_time_span(new_value)
    dpg.set_value(TIME_SPAN_INPUT, new_value)
    update_time_span_buttons()
    update_all_plots()


def on_time_span_decrement():
    """Decrement time span by step amount."""
    new_value = max(0.5, app_state.time_span - 0.5)
    app_state.set_time_span(new_value)
    dpg.set_value(TIME_SPAN_INPUT, new_value)
    update_time_span_buttons()
    update_all_plots()


def on_time_span_input_changed(sender, value):
    """Callback for time span text input."""
    app_state.set_time_span(max(0.5, min(120.0, value)))
    dpg.set_value(TIME_SPAN_INPUT, app_state.time_span)
    update_time_span_buttons()
    update_all_plots()


def on_frequency_changed(sender, value):
    """Callback for frequency slider."""
    waveform = app_state.get_active_waveform()
    if waveform:
        waveform.frequency = max(1.0, min(100.0, value))
        update_all_plots()
        update_waveform_list()


def on_amplitude_changed(sender, value):
    """Callback for amplitude slider."""
    waveform = app_state.get_active_waveform()
    if waveform:
        waveform.amplitude = max(0.0, min(10.0, value))
        update_all_plots()


def on_duty_cycle_changed(sender, value):
    """Callback for duty cycle slider."""
    waveform = app_state.get_active_waveform()
    if waveform:
        waveform.duty_cycle = max(1.0, min(100.0, value))
        update_all_plots()


def on_frequency_increment():
    """Increment frequency by step amount."""
    waveform = app_state.get_active_waveform()
    if waveform:
        new_value = min(100.0, waveform.frequency + 0.1)
        waveform.frequency = new_value
        dpg.set_value(FREQ_INPUT, new_value)
        update_waveform_parameters()
        update_all_plots()
        update_waveform_list()


def on_frequency_decrement():
    """Decrement frequency by step amount."""
    waveform = app_state.get_active_waveform()
    if waveform:
        new_value = max(1.0, waveform.frequency - 0.1)
        waveform.frequency = new_value
        dpg.set_value(FREQ_INPUT, new_value)
        update_waveform_parameters()
        update_all_plots()
        update_waveform_list()


def on_frequency_input_changed(sender, value):
    """Callback for frequency text input."""
    waveform = app_state.get_active_waveform()
    if waveform:
        waveform.frequency = max(1.0, min(100.0, value))
        dpg.set_value(FREQ_INPUT, waveform.frequency)
        update_waveform_parameters()
        update_all_plots()
        update_waveform_list()


def on_amplitude_increment():
    """Increment amplitude by step amount."""
    waveform = app_state.get_active_waveform()
    if waveform:
        new_value = min(10.0, waveform.amplitude + 0.1)
        waveform.amplitude = new_value
        dpg.set_value(AMP_INPUT, new_value)
        update_waveform_parameters()
        update_all_plots()


def on_amplitude_decrement():
    """Decrement amplitude by step amount."""
    waveform = app_state.get_active_waveform()
    if waveform:
        new_value = max(0.0, waveform.amplitude - 0.1)
        waveform.amplitude = new_value
        dpg.set_value(AMP_INPUT, new_value)
        update_waveform_parameters()
        update_all_plots()


def on_amplitude_input_changed(sender, value):
    """Callback for amplitude text input."""
    waveform = app_state.get_active_waveform()
    if waveform:
        waveform.amplitude = max(0.0, min(10.0, value))
        dpg.set_value(AMP_INPUT, waveform.amplitude)
        update_waveform_parameters()
        update_all_plots()


def on_duty_cycle_increment():
    """Increment duty cycle by step amount."""
    waveform = app_state.get_active_waveform()
    if waveform:
        new_value = min(100.0, waveform.duty_cycle + 1.0)
        waveform.duty_cycle = new_value
        dpg.set_value(DUTY_INPUT, new_value)
        update_waveform_parameters()
        update_all_plots()


def on_duty_cycle_decrement():
    """Decrement duty cycle by step amount."""
    waveform = app_state.get_active_waveform()
    if waveform:
        new_value = max(1.0, waveform.duty_cycle - 1.0)
        waveform.duty_cycle = new_value
        dpg.set_value(DUTY_INPUT, new_value)
        update_waveform_parameters()
        update_all_plots()


def on_duty_cycle_input_changed(sender, value):
    """Callback for duty cycle text input."""
    waveform = app_state.get_active_waveform()
    if waveform:
        waveform.duty_cycle = max(1.0, min(100.0, value))
        dpg.set_value(DUTY_INPUT, waveform.duty_cycle)
        update_waveform_parameters()
        update_all_plots()


def on_wave_type_changed(sender, value):
    """Callback for wave type combo box."""
    waveform = app_state.get_active_waveform()
    if waveform:
        waveform.wave_type = value.lower()
        update_waveform_parameters()
        update_all_plots()
        update_waveform_list()


def on_show_grid_changed(sender, value):
    """Callback for show grid checkbox."""
    app_state.show_grid = value
    # Update plot axis grid visibility
    x_axis = dpg.get_item_children(PLOT_TAG, slot=1)[0]  # Get X axis
    dpg.configure_item(x_axis, no_gridlines=not value)
    dpg.configure_item(Y_AXIS_TAG, no_gridlines=not value)


def on_reset_view():
    """Reset the plot view to default time span and axis limits."""
    # Reset time span to default
    app_state.set_time_span(DEFAULT_TIME_SPAN)
    dpg.set_value(TIME_SPAN_INPUT, DEFAULT_TIME_SPAN)
    update_time_span_buttons()

    # Get X axis
    x_axis = dpg.get_item_children(PLOT_TAG, slot=1)[0]

    # Reset axis limits to default and disable constraints
    dpg.set_axis_limits(x_axis, 0, DEFAULT_TIME_SPAN)
    dpg.set_axis_limits(Y_AXIS_TAG, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    # Fit axes to show full range
    dpg.set_axis_limits_auto(x_axis)
    dpg.set_axis_limits_auto(Y_AXIS_TAG)

    # Then set them back to fixed values
    dpg.set_axis_limits(x_axis, 0, DEFAULT_TIME_SPAN)
    dpg.set_axis_limits(Y_AXIS_TAG, DEFAULT_Y_MIN, DEFAULT_Y_MAX)

    # Update plots
    update_all_plots()


def on_max_envelope_changed(sender, value):
    """Callback for max envelope checkbox."""
    app_state.show_max_envelope = value
    update_envelope_controls()
    update_all_plots()


def on_min_envelope_changed(sender, value):
    """Callback for min envelope checkbox."""
    app_state.show_min_envelope = value
    update_envelope_controls()
    update_all_plots()


def on_hide_source_changed(sender, value):
    """Callback for hide source waveforms checkbox."""
    app_state.hide_source_waveforms = value
    # Enable/disable waveform management controls
    update_waveform_management_controls()
    update_all_plots()


def update_waveform_management_controls():
    """Enable/disable waveform management controls based on hide_source state."""
    global disabled_button_theme

    enabled = not app_state.hide_source_waveforms
    dpg.configure_item(ADD_WAVE_BTN, enabled=enabled)

    # Apply or remove disabled theme
    if enabled:
        dpg.bind_item_theme(ADD_WAVE_BTN, 0)
    else:
        dpg.bind_item_theme(ADD_WAVE_BTN, disabled_button_theme)

    # Update remove buttons visibility/state in waveform list
    update_waveform_list()


def handle_splitter_drag():
    """Handle mouse drag on the splitter to resize sidebar."""
    global sidebar_width, is_dragging_splitter

    # Change cursor when hovering over splitter
    if dpg.is_item_hovered(SPLITTER):
        dpg.configure_item(SPLITTER, border=True)
    else:
        dpg.configure_item(SPLITTER, border=False)

    if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
        if dpg.is_item_hovered(SPLITTER) or is_dragging_splitter:
            is_dragging_splitter = True
            mouse_pos = dpg.get_mouse_pos(local=False)
            new_width = mouse_pos[0] - dpg.get_item_pos(MAIN_WINDOW)[0]

            # Clamp width between reasonable bounds
            new_width = max(200, min(600, new_width))

            if new_width != sidebar_width:
                sidebar_width = new_width
                dpg.configure_item(SIDEBAR_CHILD, width=sidebar_width)
    else:
        is_dragging_splitter = False


def on_add_waveform():
    """Callback for add waveform button."""
    new_waveform = app_state.add_waveform()
    if new_waveform:
        update_waveform_list()
        update_waveform_parameters()
        update_envelope_controls()
        update_all_plots()
        update_add_button()


def on_remove_waveform(waveform_id: int):
    """
    Callback for remove waveform button.

    Args:
        waveform_id: ID of waveform to remove
    """
    if app_state.remove_waveform(waveform_id):
        update_waveform_list()
        update_waveform_parameters()
        update_envelope_controls()
        update_all_plots()
        update_add_button()


def on_toggle_waveform(waveform_id: int):
    """
    Callback for toggle waveform visibility.

    Args:
        waveform_id: ID of waveform to toggle
    """
    waveform = app_state.get_waveform(waveform_id)
    if waveform:
        waveform.enabled = not waveform.enabled
        update_envelope_controls()
        update_all_plots()
        update_waveform_list()


def on_select_waveform(waveform_id: int):
    """
    Callback for selecting a waveform.

    Args:
        waveform_id: ID of waveform to select
    """
    app_state.active_waveform_index = waveform_id
    update_waveform_parameters()
    update_waveform_list()


def on_export_button_clicked():
    """Callback for export button - shows file dialog."""
    dpg.show_item(FILE_DIALOG)


def on_file_dialog_ok(sender, app_data, user_data):
    """Callback when file dialog OK is clicked - performs the export."""
    # app_data contains file path information
    import os

    # Debug: Print app_data to see what we're getting
    print("\n=== FILE DIALOG DEBUG ===")
    print("File dialog app_data keys:", app_data.keys())
    for key, value in app_data.items():
        print(f"  {key}: {repr(value)}")

    # Get the selected file path
    filename = None

    if 'selections' in app_data and app_data['selections']:
        # selections is a dict of {filename: full_path}
        filename = list(app_data['selections'].values())[0]
        print(f"Using selections: {repr(filename)}")
    elif 'file_path_name' in app_data:
        filename = app_data['file_path_name']
        print(f"Using file_path_name: {repr(filename)}")
    else:
        # Manual fallback: properly combine path and filename using os.path.join
        current_path = app_data.get('current_path', '')
        file_name = app_data.get('file_name', 'waveforms.csv')

        print(f"Fallback values: current_path={repr(current_path)}, file_name={repr(file_name)}")

        # Ensure file_name has .csv extension
        if not file_name.lower().endswith('.csv'):
            file_name += '.csv'
            print(f"Added .csv to file_name: {repr(file_name)}")

        # Use os.path.join which handles separators correctly
        filename = os.path.join(current_path, file_name)
        print(f"Using os.path.join result: {repr(filename)}")

    # Ensure the final path has .csv extension (DearPyGui sometimes strips it)
    if filename and not filename.lower().endswith('.csv'):
        filename += '.csv'
        print(f"Added .csv extension to final path: {repr(filename)}")

    # Fix DearPyGui stripping the dot before extension (e.g., "waveformscsv.csv" -> "waveforms.csv")
    # This happens when user types "waveforms.csv" and DearPyGui strips the dot
    if filename and filename.lower().endswith('csv.csv'):
        filename = filename[:-7] + '.csv'  # Remove "csv.csv" and add ".csv"
        print(f"Fixed duplicate extension pattern: {repr(filename)}")

    print(f"FINAL PATH TO USE: {repr(filename)}")
    print("=========================\n")

    # Collect enabled waveform data
    waveforms_to_export = []
    for waveform in app_state.get_enabled_waveforms():
        time, amplitude = generate_waveform(
            waveform.wave_type,
            waveform.frequency,
            waveform.amplitude,
            waveform.duty_cycle,
            app_state.time_span,
            app_state.sample_rate
        )

        name = f"Waveform_{waveform.id + 1}_{waveform.wave_type.capitalize()}"
        waveform_data = prepare_waveform_for_export(
            name, time, amplitude,
            waveform.wave_type,
            waveform.frequency,
            waveform.amplitude,
            waveform.duty_cycle
        )
        waveforms_to_export.append(waveform_data)

    # Collect envelope data if enabled
    envelopes_to_export = []
    if app_state.can_show_envelopes():
        waveform_data = []
        for waveform in app_state.get_enabled_waveforms():
            time, amplitude = generate_waveform(
                waveform.wave_type,
                waveform.frequency,
                waveform.amplitude,
                waveform.duty_cycle,
                app_state.time_span,
                app_state.sample_rate
            )
            waveform_data.append((time, amplitude))

        if app_state.show_max_envelope:
            time, max_env = compute_max_envelope(waveform_data)
            envelopes_to_export.append(("Max_Envelope", time, max_env))

        if app_state.show_min_envelope:
            time, min_env = compute_min_envelope(waveform_data)
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
    dpg.set_value(EXPORT_STATUS, message)


def update_waveform_list():
    """Update the waveform list UI."""
    global toggle_on_theme, toggle_off_theme, waveform_selected_theme, waveform_unselected_theme, disabled_button_theme

    if dpg.does_item_exist(WAVEFORM_LIST_GROUP):
        dpg.delete_item(WAVEFORM_LIST_GROUP, children_only=True)

        for waveform in app_state.waveforms:
            with dpg.group(horizontal=True, parent=WAVEFORM_LIST_GROUP):
                # Waveform info button
                label = f"Waveform {waveform.id + 1}"

                waveform_btn = dpg.add_button(
                    label=label,
                    width=180,
                    callback=lambda s, a, u: on_select_waveform(u),
                    user_data=waveform.id
                )

                # Apply selection theme
                is_selected = waveform.id == app_state.active_waveform_index
                if is_selected:
                    dpg.bind_item_theme(waveform_btn, waveform_selected_theme)
                else:
                    dpg.bind_item_theme(waveform_btn, waveform_unselected_theme)

                # Toggle visibility button
                waveform_visibility = "ON" if waveform.enabled else "OFF"
                toggle_btn = dpg.add_button(
                    label=waveform_visibility,
                    width=40,
                    callback=lambda s, a, u: on_toggle_waveform(u),
                    user_data=waveform.id
                )

                # Apply theme based on state
                if waveform.enabled:
                    dpg.bind_item_theme(toggle_btn, toggle_on_theme)
                else:
                    dpg.bind_item_theme(toggle_btn, toggle_off_theme)

                # Remove button (hidden if at min waveforms or hide_source is active)
                can_remove = len(app_state.waveforms) > app_state.MIN_WAVEFORMS
                is_enabled = not app_state.hide_source_waveforms
                if can_remove:
                    remove_btn = dpg.add_button(
                        label="X",
                        width=30,
                        callback=lambda s, a, u: on_remove_waveform(u),
                        user_data=waveform.id,
                        enabled=is_enabled
                    )
                    # Apply disabled theme if not enabled
                    if not is_enabled:
                        dpg.bind_item_theme(remove_btn, disabled_button_theme)
                else:
                    # Add spacer button (invisible) to maintain alignment
                    dpg.add_button(label="", width=30, enabled=False, show=False)


def update_time_span_buttons():
    """Update time span button states based on current value."""
    global disabled_button_theme

    time_at_min = app_state.time_span <= 0.5
    time_at_max = app_state.time_span >= 120.0
    dpg.configure_item(TIME_SPAN_DEC_BTN, enabled=not time_at_min)
    dpg.configure_item(TIME_SPAN_INC_BTN, enabled=not time_at_max)
    dpg.bind_item_theme(TIME_SPAN_DEC_BTN, disabled_button_theme if time_at_min else 0)
    dpg.bind_item_theme(TIME_SPAN_INC_BTN, disabled_button_theme if time_at_max else 0)


def update_waveform_parameters():
    """Update waveform parameter inputs based on active waveform."""
    global disabled_button_theme

    waveform = app_state.get_active_waveform()
    if not waveform:
        return

    # Update input fields
    dpg.set_value(FREQ_INPUT, waveform.frequency)
    dpg.set_value(AMP_INPUT, waveform.amplitude)
    dpg.set_value(DUTY_INPUT, waveform.duty_cycle)
    dpg.set_value(WAVE_TYPE_COMBO, waveform.wave_type.capitalize())

    # Update frequency button states
    freq_at_min = waveform.frequency <= 1.0
    freq_at_max = waveform.frequency >= 100.0
    dpg.configure_item(FREQ_DEC_BTN, enabled=not freq_at_min)
    dpg.configure_item(FREQ_INC_BTN, enabled=not freq_at_max)
    dpg.bind_item_theme(FREQ_DEC_BTN, disabled_button_theme if freq_at_min else 0)
    dpg.bind_item_theme(FREQ_INC_BTN, disabled_button_theme if freq_at_max else 0)

    # Update amplitude button states
    amp_at_min = waveform.amplitude <= 0.0
    amp_at_max = waveform.amplitude >= 10.0
    dpg.configure_item(AMP_DEC_BTN, enabled=not amp_at_min)
    dpg.configure_item(AMP_INC_BTN, enabled=not amp_at_max)
    dpg.bind_item_theme(AMP_DEC_BTN, disabled_button_theme if amp_at_min else 0)
    dpg.bind_item_theme(AMP_INC_BTN, disabled_button_theme if amp_at_max else 0)

    # Update duty cycle button states
    duty_at_min = waveform.duty_cycle <= 1.0
    duty_at_max = waveform.duty_cycle >= 100.0
    dpg.configure_item(DUTY_DEC_BTN, enabled=not duty_at_min)
    dpg.configure_item(DUTY_INC_BTN, enabled=not duty_at_max)
    dpg.bind_item_theme(DUTY_DEC_BTN, disabled_button_theme if duty_at_min else 0)
    dpg.bind_item_theme(DUTY_INC_BTN, disabled_button_theme if duty_at_max else 0)

    # Show/hide duty cycle for square waves only
    needs_duty = waveform.wave_type.lower() == 'square'
    dpg.configure_item(DUTY_LABEL, show=needs_duty)
    dpg.configure_item(DUTY_GROUP, show=needs_duty)


def update_envelope_controls():
    """Enable/disable envelope checkboxes based on number of enabled waveforms."""
    global disabled_checkbox_theme

    can_show = app_state.can_show_envelopes()

    # Update max envelope checkbox
    dpg.configure_item(MAX_ENV_CHECKBOX, enabled=can_show)
    if can_show:
        dpg.bind_item_theme(MAX_ENV_CHECKBOX, 0)
        dpg.configure_item(MAX_ENV_LABEL, color=ENABLED_LABEL_COLOR)
    else:
        dpg.bind_item_theme(MAX_ENV_CHECKBOX, disabled_checkbox_theme)
        dpg.configure_item(MAX_ENV_LABEL, color=DISABLED_LABEL_COLOR)

    # Update min envelope checkbox
    dpg.configure_item(MIN_ENV_CHECKBOX, enabled=can_show)
    if can_show:
        dpg.bind_item_theme(MIN_ENV_CHECKBOX, 0)
        dpg.configure_item(MIN_ENV_LABEL, color=ENABLED_LABEL_COLOR)
    else:
        dpg.bind_item_theme(MIN_ENV_CHECKBOX, disabled_checkbox_theme)
        dpg.configure_item(MIN_ENV_LABEL, color=DISABLED_LABEL_COLOR)

    # Hide source checkbox requires at least one envelope to be shown
    can_hide_source = can_show and (app_state.show_max_envelope or app_state.show_min_envelope)
    dpg.configure_item(HIDE_SOURCE_CHECKBOX, enabled=can_hide_source)

    # Apply or remove theme based on enabled state
    if can_hide_source:
        # Remove theme when enabled (use default theme)
        dpg.bind_item_theme(HIDE_SOURCE_CHECKBOX, 0)
        dpg.configure_item(HIDE_SOURCE_LABEL, color=ENABLED_LABEL_COLOR)
    else:
        # Apply disabled theme when disabled
        dpg.bind_item_theme(HIDE_SOURCE_CHECKBOX, disabled_checkbox_theme)
        dpg.configure_item(HIDE_SOURCE_LABEL, color=DISABLED_LABEL_COLOR)

    # If hide source becomes disabled, turn it off and uncheck it
    if not can_hide_source:
        if app_state.hide_source_waveforms:
            app_state.hide_source_waveforms = False
            update_waveform_management_controls()
        dpg.set_value(HIDE_SOURCE_CHECKBOX, False)

    if not can_show:
        app_state.show_max_envelope = False
        app_state.show_min_envelope = False
        dpg.set_value(MAX_ENV_CHECKBOX, False)
        dpg.set_value(MIN_ENV_CHECKBOX, False)


def update_add_button():
    """Enable/disable add waveform button based on max limit."""
    global disabled_button_theme

    can_add = len(app_state.waveforms) < app_state.MAX_WAVEFORMS
    dpg.configure_item(ADD_WAVE_BTN, enabled=can_add)

    # Apply or remove disabled theme
    if can_add:
        dpg.bind_item_theme(ADD_WAVE_BTN, 0)
    else:
        dpg.bind_item_theme(ADD_WAVE_BTN, disabled_button_theme)


def update_status_bar():
    """Update status bar with current info."""
    num_waveforms = len(app_state.waveforms)
    fps = dpg.get_frame_rate()
    status_text = f"Sample Rate: {app_state.sample_rate} S/s | FPS: {fps:.0f} | Waveforms: {num_waveforms}/{app_state.MAX_WAVEFORMS}"
    dpg.set_value(STATUS_BAR, status_text)


def create_ui():
    """Create the main UI layout."""
    global toggle_on_theme, toggle_off_theme, disabled_checkbox_theme, disabled_button_theme, waveform_selected_theme, waveform_unselected_theme

    # Create toggle button themes
    toggle_on_theme, toggle_off_theme = create_toggle_button_themes()

    # Create waveform button themes
    waveform_selected_theme, waveform_unselected_theme = create_waveform_button_themes()

    # Create disabled checkbox theme
    disabled_checkbox_theme = create_disabled_checkbox_theme()

    # Create disabled button theme
    disabled_button_theme = create_disabled_button_theme()

    with dpg.window(label="Waveform Generator/Analyzer", tag=MAIN_WINDOW, width=1200, height=800):
        with dpg.group(horizontal=True):
            # Sidebar
            with dpg.child_window(width=sidebar_width, height=-25, tag=SIDEBAR_CHILD):
                # Display Controls
                dpg.add_text("Display Controls", color=(255, 255, 0))
                dpg.add_separator()

                # Time Span control with +/- buttons
                dpg.add_text("Time Span (s)")
                with dpg.group(horizontal=True, tag=TIME_SPAN_GROUP):
                    dpg.add_input_float(
                        tag=TIME_SPAN_INPUT,
                        default_value=DEFAULT_TIME_SPAN,
                        min_value=0.5,
                        max_value=120.0,
                        min_clamped=True,
                        max_clamped=True,
                        callback=on_time_span_input_changed,
                        on_enter=True,
                        step=0.0,
                        width=134,
                        format="%.1f"
                    )
                    dpg.add_button(
                        label="-",
                        tag=TIME_SPAN_DEC_BTN,
                        callback=on_time_span_decrement,
                        width=30
                    )
                    dpg.add_button(
                        label="+",
                        tag=TIME_SPAN_INC_BTN,
                        callback=on_time_span_increment,
                        width=30
                    )

                dpg.add_checkbox(
                    label="Show Grid",
                    tag=SHOW_GRID_CHECKBOX,
                    default_value=True,
                    callback=on_show_grid_changed
                )

                dpg.add_button(
                    label="Reset View",
                    callback=on_reset_view,
                    width=-1
                )

                dpg.add_separator()

                # Advanced
                dpg.add_text("Advanced", color=(255, 255, 0))
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    max_env_cb = dpg.add_checkbox(
                        tag=MAX_ENV_CHECKBOX,
                        default_value=False,
                        callback=on_max_envelope_changed,
                        enabled=False
                    )
                    dpg.bind_item_theme(max_env_cb, disabled_checkbox_theme)
                    dpg.add_text("Show Max Envelope", tag=MAX_ENV_LABEL, color=DISABLED_LABEL_COLOR)

                with dpg.group(horizontal=True):
                    min_env_cb = dpg.add_checkbox(
                        tag=MIN_ENV_CHECKBOX,
                        default_value=False,
                        callback=on_min_envelope_changed,
                        enabled=False
                    )
                    dpg.bind_item_theme(min_env_cb, disabled_checkbox_theme)
                    dpg.add_text("Show Min Envelope", tag=MIN_ENV_LABEL, color=DISABLED_LABEL_COLOR)

                with dpg.group(horizontal=True):
                    hide_src_cb = dpg.add_checkbox(
                        tag=HIDE_SOURCE_CHECKBOX,
                        default_value=False,
                        callback=on_hide_source_changed,
                        enabled=False
                    )
                    dpg.bind_item_theme(hide_src_cb, disabled_checkbox_theme)
                    dpg.add_text("Hide Source Waveforms", tag=HIDE_SOURCE_LABEL, color=DISABLED_LABEL_COLOR)

                dpg.add_separator()

                # Waveforms List
                dpg.add_text("Waveforms", color=(255, 255, 0))
                dpg.add_separator()

                dpg.add_button(
                    label="+ Add Waveform",
                    tag=ADD_WAVE_BTN,
                    callback=on_add_waveform,
                    width=-1
                )

                dpg.add_group(tag=WAVEFORM_LIST_GROUP)

                dpg.add_separator()

                # Waveform Parameters
                dpg.add_text("Waveform Parameters", color=(255, 255, 0))
                dpg.add_separator()

                dpg.add_combo(
                    label="Type",
                    tag=WAVE_TYPE_COMBO,
                    items=["Sine", "Square", "Sawtooth", "Triangle"],
                    default_value="Sine",
                    callback=on_wave_type_changed,
                    width=200
                )

                # Frequency control with +/- buttons
                dpg.add_text("Frequency (Hz)")
                with dpg.group(horizontal=True, tag=FREQ_GROUP):
                    dpg.add_input_float(
                        tag=FREQ_INPUT,
                        default_value=1.0,
                        min_value=1.0,
                        max_value=100.0,
                        min_clamped=True,
                        max_clamped=True,
                        callback=on_frequency_input_changed,
                        on_enter=True,
                        step=0.0,
                        width=134,
                        format="%.1f"
                    )
                    dpg.add_button(
                        label="-",
                        tag=FREQ_DEC_BTN,
                        callback=on_frequency_decrement,
                        width=30
                    )
                    dpg.add_button(
                        label="+",
                        tag=FREQ_INC_BTN,
                        callback=on_frequency_increment,
                        width=30
                    )

                # Amplitude control with +/- buttons
                dpg.add_text("Amplitude")
                with dpg.group(horizontal=True, tag=AMP_GROUP):
                    dpg.add_input_float(
                        tag=AMP_INPUT,
                        default_value=5.0,
                        min_value=0.0,
                        max_value=10.0,
                        min_clamped=True,
                        max_clamped=True,
                        callback=on_amplitude_input_changed,
                        on_enter=True,
                        step=0.0,
                        width=134,
                        format="%.1f"
                    )
                    dpg.add_button(
                        label="-",
                        tag=AMP_DEC_BTN,
                        callback=on_amplitude_decrement,
                        width=30
                    )
                    dpg.add_button(
                        label="+",
                        tag=AMP_INC_BTN,
                        callback=on_amplitude_increment,
                        width=30
                    )

                # Duty Cycle control with +/- buttons
                dpg.add_text("Duty Cycle (%)", tag=DUTY_LABEL, show=False)
                with dpg.group(horizontal=True, tag=DUTY_GROUP, show=False):
                    dpg.add_input_float(
                        tag=DUTY_INPUT,
                        default_value=50.0,
                        min_value=1.0,
                        max_value=100.0,
                        min_clamped=True,
                        max_clamped=True,
                        callback=on_duty_cycle_input_changed,
                        on_enter=True,
                        step=0.0,
                        width=134,
                        format="%.1f"
                    )
                    dpg.add_button(
                        label="-",
                        tag=DUTY_DEC_BTN,
                        callback=on_duty_cycle_decrement,
                        width=30
                    )
                    dpg.add_button(
                        label="+",
                        tag=DUTY_INC_BTN,
                        callback=on_duty_cycle_increment,
                        width=30
                    )

                dpg.add_separator()

                # Export
                dpg.add_text("Export", color=(255, 255, 0))
                dpg.add_separator()

                dpg.add_button(
                    label="Export to CSV",
                    tag=EXPORT_BTN,
                    callback=on_export_button_clicked,
                    width=-1
                )

                dpg.add_text("Status: Ready", tag=EXPORT_STATUS, color=(0, 255, 0))

            # Draggable splitter
            with dpg.child_window(width=4, height=-25, tag=SPLITTER):
                pass

            # Main plot area
            with dpg.child_window(height=-25):
                with dpg.plot(label="Waveforms", height=-1, width=-1, tag=PLOT_TAG):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", no_gridlines=not app_state.show_grid)
                    dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag=Y_AXIS_TAG, no_gridlines=not app_state.show_grid)

        # Status bar
        dpg.add_text("", tag=STATUS_BAR)

    # Create file dialog for export (save mode)
    # Note: DearPyGui strips dots from default_filename, so we use "waveforms"
    # and rely on the file extension filter to add ".csv"
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=on_file_dialog_ok,
        tag=FILE_DIALOG,
        width=700,
        height=400,
        default_filename="waveforms",
        file_count=1,
        modal=True
    ):
        dpg.add_file_extension(".csv", color=(0, 255, 0, 255), custom_text="[CSV]")

    # Register mouse handler for splitter
    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=lambda: handle_splitter_drag())

    # Initialize UI state
    update_time_span_buttons()
    update_waveform_list()
    update_waveform_parameters()
    update_envelope_controls()
    update_add_button()
    update_all_plots()
