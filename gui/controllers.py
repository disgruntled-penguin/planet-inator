import pygame
import pygame_gui as pg

import numpy as np

class PygameGUIControls:
    def __init__(self, sim, viewport, paused_ref, stars_visible_ref):
        self.sim = sim
        self.viewport = viewport
        self.paused_ref = paused_ref  
        self.stars_visible_ref = stars_visible_ref  # Same as above
        self.manager = pg.UIManager((1000, 1000))
        self.panel_collapsed = False  
        self.param_panel_collapsed = False  
        self.last_slider_value = 1.0  
        self._setup_ui()

    def _setup_ui(self):
        
        
        #self.manager.set_theme(pg.themes.THEME_DARK)
        self.manager = pg.UIManager((1300, 1000), 'gui/theme.json')
        print(f"[debug] UI Manager initialized with size: 1300x1000")
       # self.controls = pg.elements.UIDropDownMenu(relative_rect=pygame.Rect(200, 100, 200, 30), manager=self.manager, options_list=['Pause', 'Trev', 'Bob'],starting_option='Pause')
        self.panel = pg.elements.UIPanel(relative_rect=pygame.Rect((0, 0), (250, 200)), manager=self.manager)
       
        self.dropdown_button = pg.elements.UIButton(relative_rect=pygame.Rect((50, 0), (100, 15)), text='▼ Controls', manager=self.manager, container=self.panel)
       
        self.pause_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 30), (200, 30)), text='Pause (P)', manager=self.manager, container=self.panel)
        self.zoom_in_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 50), (200, 30)), text='Zoom In (+ or Scroll down)', manager=self.manager, container=self.panel)
        self.zoom_out_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 70), (200, 30)), text='Zoom Out (- or Scroll up)', manager=self.manager, container=self.panel)
        self.reset_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 90), (200, 30)), text='Reset (R)', manager=self.manager, container=self.panel)
        self.stars_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 110), (200, 30)), text='Toggle Stars (T)', manager=self.manager, container=self.panel)
        self.screenshot_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 130), (200, 30)), text='Screenshot (S)', manager=self.manager, container=self.panel)
        self.quit_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 150), (200, 30)), text='Quit (Q)', manager=self.manager, container=self.panel)
        

        # Input fields for Doof's parameters - moved to bottom right
        self.param_panel = pg.elements.UIPanel(relative_rect=pygame.Rect((1000, 580), (250, 210)), manager=self.manager)
        
       
        self.dropdown_param_button = pg.elements.UIButton(relative_rect=pygame.Rect((50, 0), (150, 15)), text='▼ Doof Parameters', manager=self.manager, container=self.param_panel)

        self.a_input = pg.elements.UITextEntryLine(relative_rect=pygame.Rect((10, 30), (100, 30)), manager=self.manager, container=self.param_panel)
        self.a_input.set_text("3.3")
        self.a_label = pg.elements.UILabel(relative_rect=pygame.Rect((120, 30), (100, 30)), text="(AU)", manager=self.manager, container=self.param_panel)

        self.e_input = pg.elements.UITextEntryLine(relative_rect=pygame.Rect((10, 70), (100, 30)), manager=self.manager, container=self.param_panel)
        self.e_input.set_text("0.3")
        self.e_label = pg.elements.UILabel(relative_rect=pygame.Rect((120, 70), (100, 30)), text="ecc", manager=self.manager, container=self.param_panel)

        self.inc_input = pg.elements.UITextEntryLine(relative_rect=pygame.Rect((10, 110), (100, 30)), manager=self.manager, container=self.param_panel)
        self.inc_input.set_text("15")
        self.inc_label = pg.elements.UILabel(relative_rect=pygame.Rect((120, 110), (100, 30)), text="inc (°)", manager=self.manager, container=self.param_panel)

        self.mass_input = pg.elements.UITextEntryLine(relative_rect=pygame.Rect((10, 150), (100, 30)), manager=self.manager, container=self.param_panel)
        self.mass_input.set_text("400")
        self.mass_label = pg.elements.UILabel(relative_rect=pygame.Rect((120, 150), (100, 30)), text="mass (×m⊕)", manager=self.manager, container=self.param_panel)
        

        self.doof_submit = pg.elements.UIButton(relative_rect=pygame.Rect((10, 180), (200, 30)), text="Update Doof", manager=self.manager, container=self.param_panel)
        # Speed control slider - try without panel first to test visibility
        self.speed_label = pg.elements.UILabel(relative_rect=pygame.Rect((500, 400), (80, 30)), text="Speed:", manager=self.manager)
        
        self.speed_slider = pg.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((550, 780), (100, 50)),
            start_value=1.0, 
            value_range=(1, 1000.0),  
            manager=self.manager,
            click_increment=10  
        )
        
        self.speed_value_label = pg.elements.UILabel(relative_rect=pygame.Rect((650, 780), (140, 50)), text="1 year/sec", manager=self.manager)
        
        print(f"[debug] Speed slider created at position: {self.speed_slider.relative_rect}")
        print(f"[debug] Speed slider visibility: {self.speed_slider.visible}")
        print(f"[debug] Speed slider current value: {self.speed_slider.get_current_value()}")

        self._set_panel_visibility(False)
        self._set_param_panel_visibility(False)
        

      
      
        
    def _set_panel_visibility(self, visible):
        
        for btn in [self.pause_button, self.zoom_in_button, self.zoom_out_button, self.reset_button, self.stars_button, self.screenshot_button, self.quit_button]:
            btn.visible = visible
       
        if visible:
            self.panel.set_relative_position((10, 10))
            self.panel.set_dimensions((250, 210))
            self.dropdown_button.set_text('hide controls')
        else:
            self.panel.set_relative_position((10, 10))
            self.panel.set_dimensions((250, 20))
            self.dropdown_button.set_text('controls')

    def _set_param_panel_visibility(self, visible):
        
        for element in [self.a_input, self.a_label, self.e_input, self.e_label, self.inc_input, self.inc_label, self.mass_input, self.mass_label, self.doof_submit]:
            element.visible = visible
        # Resize panel
        if visible:
            self.param_panel.set_relative_position((1000, 580))
            self.param_panel.set_dimensions((250, 210))
            self.dropdown_param_button.set_text('hide parameters')
        else:
            self.param_panel.set_relative_position((1000, 780))
            self.param_panel.set_dimensions((250, 20))
            self.dropdown_param_button.set_text('Doof Parameters')

    def draw_ui(self, screen):
        self.manager.draw_ui(screen)

    def process_events(self, event, screen, save_screenshot, output_video, video_writer):
        self.manager.process_events(event)
        if event.type == pygame.USEREVENT:
            if event.user_type == pg.UI_BUTTON_PRESSED:
                if event.ui_element == self.dropdown_button:
                    self.panel_collapsed = not self.panel_collapsed
                    self._set_panel_visibility(not self.panel_collapsed)
                elif event.ui_element == self.dropdown_param_button:
                    self.param_panel_collapsed = not self.param_panel_collapsed
                    self._set_param_panel_visibility(not self.param_panel_collapsed)
                elif not self.panel_collapsed:
                    if event.ui_element == self.pause_button:
                        self.paused_ref['value'] = not self.paused_ref['value']
                        self.pause_button.set_text('Resume' if self.paused_ref['value'] else 'Pause')
                    elif event.ui_element == self.zoom_in_button:
                        self.viewport.zoom_in()
                    elif event.ui_element == self.zoom_out_button:
                        self.viewport.zoom_out()
                    elif event.ui_element == self.reset_button:
                        self.viewport.reset_zoom()
                    elif event.ui_element == self.stars_button:
                        self.stars_visible_ref['value'] = not self.stars_visible_ref['value']
                    elif event.ui_element == self.screenshot_button:
                        save_screenshot(screen, self.sim.t)
                    elif event.ui_element == self.quit_button:
                        if output_video:
                            video_writer.release()
                        pygame.quit()
                        quit()
                        
                
                if event.ui_element == self.doof_submit:
                    try:
                        print("[debug] Submit button pressed.")

                        
                        a = float(self.a_input.get_text())
                        e = float(self.e_input.get_text())
                        inc_deg = float(self.inc_input.get_text())
                        m = float(self.mass_input.get_text())

                        new_params = {
                            "a": a,
                            "e": e,
                            "inc": np.radians(inc_deg),
                            "m": m * (1 / 333000),  # Earth mass to Solar mass
                        }

                        self.sim.update_doof_params(new_params)
                        print("[✓] Doof's Planet updated.")
                    except ValueError:
                        print("[!] Invalid input for Doof's Planet update.")

          
            elif event.user_type == pg.UI_HORIZONTAL_SLIDER_MOVED:
                print(f"[debug] Slider moved event detected!")
                if event.ui_element == self.speed_slider:
                    speed_multiplier = self.speed_slider.get_current_value()
                    # Base delta_t = (14 * np.pi) / 100
                    base_delta_t = (14 * np.pi) / 100
                    self.sim.delta_t = base_delta_t * speed_multiplier
                    
                    self.speed_value_label.set_text(f"{speed_multiplier:.1f} years/second")
                    print(f"[debug] Speed set to {speed_multiplier:.1f}x (delta_t = {self.sim.delta_t:.4f})")
                else:
                    print(f"[debug] Slider event but not our slider: {event.ui_element}")
            
            elif event.user_type in [pg.UI_BUTTON_PRESSED, pg.UI_BUTTON_START_PRESS]:
                print(f"[debug] Button event: {event.user_type} on {event.ui_element}")
                print(f"[debug] Button parent: {getattr(event.ui_element, 'container', 'No container')}")
                
                if hasattr(event.ui_element, 'container') and event.ui_element.container == self.speed_slider:
                    print(f"[debug] This is a slider arrow button!")
                    speed_multiplier = self.speed_slider.get_current_value()
                    base_delta_t = (14 * np.pi) / 100
                    self.sim.delta_t = base_delta_t * speed_multiplier
                    self.speed_value_label.set_text(f"{speed_multiplier:.1f}yrs/sec")
                    print(f"[debug] Speed updated via arrow to {speed_multiplier:.1f}x")
                
                if event.ui_element == self.dropdown_button:
                    self.panel_collapsed = not self.panel_collapsed
                    self._set_panel_visibility(not self.panel_collapsed)
                elif event.ui_element == self.dropdown_param_button:
                    self.param_panel_collapsed = not self.param_panel_collapsed
                    self._set_param_panel_visibility(not self.param_panel_collapsed)
                elif not self.panel_collapsed:
                    if event.ui_element == self.pause_button:
                        self.paused_ref['value'] = not self.paused_ref['value']
                        self.pause_button.set_text('Resume' if self.paused_ref['value'] else 'Pause')
                    elif event.ui_element == self.zoom_in_button:
                        self.viewport.zoom_in()
                    elif event.ui_element == self.zoom_out_button:
                        self.viewport.zoom_out()
                    elif event.ui_element == self.reset_button:
                        self.viewport.reset_zoom()
                    elif event.ui_element == self.stars_button:
                        self.stars_visible_ref['value'] = not self.stars_visible_ref['value']
                    elif event.ui_element == self.screenshot_button:
                        save_screenshot(screen, self.sim.t)
                    elif event.ui_element == self.quit_button:
                        if output_video:
                            video_writer.release()
                        pygame.quit()
                        quit()
                        
                if event.ui_element == self.doof_submit:
                    try:
                        print("[debug] Submit button pressed.")

                        
                        a = float(self.a_input.get_text())
                        e = float(self.e_input.get_text())
                        inc_deg = float(self.inc_input.get_text())
                        m = float(self.mass_input.get_text())

                        new_params = {
                            "a": a,
                            "e": e,
                            "inc": np.radians(inc_deg),
                            "m": m * (1 / 333000),  # Earth mass to Solar mass
                        }

                        self.sim.update_doof_params(new_params)
                        print("[✓] Doof's Planet updated.")
                    except ValueError:
                        print("[!] Invalid input for Doof's Planet update.")
            
            # debugging
            elif event.user_type in [pg.UI_BUTTON_START_PRESS]:
                print(f"[debug] Button start press: {event.user_type} on {event.ui_element}")
                
        elif event.type == pygame.MOUSEBUTTONDOWN:
            print(f"[debug] Mouse button down at {event.pos}")
        elif event.type == pygame.MOUSEBUTTONUP:
            print(f"[debug] Mouse button up at {event.pos}")
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:  
                print(f"[debug] Mouse drag to {event.pos}")

    def update(self, time_delta):
        self.manager.update(time_delta)
        

        current_slider_value = self.speed_slider.get_current_value()
        if abs(current_slider_value - self.last_slider_value) > 0.01:  # Small threshold for float comparison
            print(f"[debug] Slider value changed from {self.last_slider_value:.2f} to {current_slider_value:.2f}")
            # updatespeed
            base_delta_t = (14 * np.pi) / 100
            self.sim.delta_t = base_delta_t * current_slider_value
            
         
            self.speed_value_label.set_text(f"{current_slider_value:.0f} years/second")
            print(f"[debug] Speed updated to {current_slider_value:.1f}x (delta_t = {self.sim.delta_t:.4f})")
            
        
            self.last_slider_value = current_slider_value