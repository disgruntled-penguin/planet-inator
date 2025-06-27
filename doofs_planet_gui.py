import dearpygui.dearpygui as dpg
import numpy as np
from doofs_planet_state import doofs_planet_state

def show_doofs_planet_gui():
    dpg.create_context()
    dpg.create_viewport(title='Doofs Planet Controls', width=800, height=600)

    # Shortcuts/actions info
    shortcut_info = {
        "Stars: T": "Toggle stars on/off (T)",
        "Zoom In: +": "Zoom in (+)",
        "Zoom Out: -": "Zoom out (-)",
        "Reset Zoom: R": "Reset zoom (R)",
        "Pause: P": "Pause simulation (P)",
        "Screenshot: S": "Save screenshot (S)",
        "Quit: Q": "Quit simulation (Q)"
    }

    with dpg.window(label="Shortcuts", pos=[10, 10], width=780, height=80, no_title_bar=True, no_resize=True, no_move=True):
        for i, (label, tip) in enumerate(shortcut_info.items()):
            tid = dpg.add_text(label, pos=[20 + i*110, 20])
            with dpg.tooltip(tid):
                dpg.add_text(tip)

    # Callbacks for updating shared state
    def mass_callback(sender, app_data):
        with doofs_planet_state.lock:
            doofs_planet_state.mass = app_data
    def size_callback(sender, app_data):
        with doofs_planet_state.lock:
            doofs_planet_state.size = app_data
    def color_callback(sender, app_data):
        with doofs_planet_state.lock:
            doofs_planet_state.color = tuple(int(x) for x in app_data)
    def a_callback(sender, app_data):
        with doofs_planet_state.lock:
            doofs_planet_state.a = app_data
    def e_callback(sender, app_data):
        with doofs_planet_state.lock:
            doofs_planet_state.e = app_data
    def inc_callback(sender, app_data):
        with doofs_planet_state.lock:
            doofs_planet_state.inc = app_data
    def Omega_callback(sender, app_data):
        with doofs_planet_state.lock:
            doofs_planet_state.Omega = app_data
    def omega_callback(sender, app_data):
        with doofs_planet_state.lock:
            doofs_planet_state.omega = app_data
    def f_callback(sender, app_data):
        with doofs_planet_state.lock:
            doofs_planet_state.f = app_data

    # Doofs planet specs (bottom left)
    with dpg.window(label="Doofs Planet Specs", pos=[10, 120], width=350, height=400):
        dpg.add_text("Edit Doofs Planet Parameters:")
        dpg.add_input_float(label="Mass (in Earth masses)", default_value=400, min_value=0.1, max_value=1000, step=1, callback=mass_callback)
        dpg.add_input_float(label="Size (display radius)", default_value=0.07, min_value=0.01, max_value=1.0, step=0.01, callback=size_callback)
        dpg.add_color_edit(label="Color", default_value=(128, 0, 128, 255), no_alpha=False, callback=color_callback)
        dpg.add_input_float(label="a (semi-major axis, AU)", default_value=3.3, min_value=0.1, max_value=50, step=0.1, callback=a_callback)
        dpg.add_input_float(label="e (eccentricity)", default_value=0.3, min_value=0.0, max_value=0.99, step=0.01, callback=e_callback)
        dpg.add_input_float(label="inc (deg)", default_value=15, min_value=0, max_value=180, step=1, callback=inc_callback)
        dpg.add_input_float(label="Omega (deg)", default_value=80, min_value=0, max_value=360, step=1, callback=Omega_callback)
        dpg.add_input_float(label="omega (deg)", default_value=60, min_value=0, max_value=360, step=1, callback=omega_callback)
        dpg.add_input_float(label="f (deg)", default_value=10, min_value=0, max_value=360, step=1, callback=f_callback)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    show_doofs_planet_gui() 