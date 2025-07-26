import pygame
import pygame_gui as pg
import numpy as np
import time
import threading
from queue import Queue
from spock import DeepRegressor

class PygameGUIControls:
    def __init__(self, sim, viewport, paused_ref, stars_visible_ref, asteroid_visibility):
        self.sim = sim
        self.viewport = viewport
        self.paused_ref = paused_ref  
        self.stars_visible_ref = stars_visible_ref
        self.asteroid_visibility = asteroid_visibility  # Add asteroid visibility reference
        self.manager = pg.UIManager((1000, 1000))
        self.panel_collapsed = False  
        self.param_panel_collapsed = False  
        self.asteroid_panel_collapsed = True  # Start collapsed
        self.last_slider_value = 1.0
        
        
        self.prediction_queue = Queue()
        self.current_prediction = None
        self.prediction_thread = None
        self.prediction_in_progress = False
        
        self.show_intro = True
        self.intro_start_time = time.time()
        self.intro_duration = 30 # 8 seconds
        self.intro_clicked = False
        self.bubble_alpha = 0  # For fade animation
        self.bubble_target_alpha = 220  # Target transparency

        self.info_bubble_visible = False
        self.info_bubble_data = None
        self.info_bubble = None
        self.info_bubble_text = None
        
        self._setup_ui()
        self._setup_intro_bubble()

    def _setup_ui(self):
        #self.manager.set_theme(pg.themes.THEME_DARK)
        self.manager = pg.UIManager((1300, 1000), 'gui/theme.json')
        
        
        self.panel = pg.elements.UIPanel(relative_rect=pygame.Rect((0, 0), (250, 240)), manager=self.manager)
       
        self.dropdown_button = pg.elements.UIButton(relative_rect=pygame.Rect((50, 0), (100, 15)), text='▼ Controls', manager=self.manager, container=self.panel)
       
        self.pause_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 30), (200, 30)), text='Pause (P)', manager=self.manager, container=self.panel)
        self.zoom_in_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 50), (200, 30)), text='Zoom In (+ or Scroll down)', manager=self.manager, container=self.panel)
        self.zoom_out_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 70), (200, 30)), text='Zoom Out (- or Scroll up)', manager=self.manager, container=self.panel)
        self.reset_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 90), (200, 30)), text='Reset (R)', manager=self.manager, container=self.panel)
        self.stars_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 110), (200, 30)), text='Toggle Stars (T)', manager=self.manager, container=self.panel)
        self.screenshot_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 130), (200, 30)), text='Screenshot (S)', manager=self.manager, container=self.panel)
        
        # Asteroid Options submenu button
        self.asteroid_options_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 160), (200, 30)), text='▶ Asteroid Options', manager=self.manager, container=self.panel)
        
        self.quit_button = pg.elements.UIButton(relative_rect=pygame.Rect((0, 190), (200, 30)), text='Quit (Q)', manager=self.manager, container=self.panel)
        
        # Asteroid Options Panel (submenu)
        self.asteroid_panel = pg.elements.UIPanel(relative_rect=pygame.Rect((260, 10), (220, 180)), manager=self.manager)
        
        self.asteroid_panel_title = pg.elements.UILabel(relative_rect=pygame.Rect((10, 5), (200, 20)), text='Asteroid Options', manager=self.manager, container=self.asteroid_panel)
        
        # NEA Controls
        self.nea_toggle_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 30), (200, 25)), text='Toggle NEA Asteroids (1)', manager=self.manager, container=self.asteroid_panel)
        self.nea_orbits_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 60), (200, 25)), text='Toggle NEA Orbits (2)', manager=self.manager, container=self.asteroid_panel)
        
        # Distant Controls
        self.distant_toggle_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 90), (200, 25)), text='Toggle Distant Asteroids (3)', manager=self.manager, container=self.asteroid_panel)
        self.distant_orbits_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 120), (200, 25)), text='Toggle Distant Orbits (4)', manager=self.manager, container=self.asteroid_panel)
        
        # All Orbits Toggle
        self.all_orbits_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 150), (200, 25)), text='Toggle All Orbits (O)', manager=self.manager, container=self.asteroid_panel)
        
# Input fields for Doof's parameters - moved to bottom right
        self.param_panel = pg.elements.UIPanel(relative_rect=pygame.Rect((1000, 430), (290, 360)), manager=self.manager)

        self.dropdown_param_button = pg.elements.UIButton(relative_rect=pygame.Rect((50, 0), (150, 15)), text='Planet-inator', manager=self.manager, container=self.param_panel)

        self.a_input = pg.elements.UITextEntryLine(relative_rect=pygame.Rect((10, 30), (100, 30)), manager=self.manager, container=self.param_panel)
        self.a_input.set_text("3.3")
        self.a_label = pg.elements.UILabel(relative_rect=pygame.Rect((120, 30), (160, 30)), text="Distance from Sun (AU)", manager=self.manager, container=self.param_panel)

        self.e_input = pg.elements.UITextEntryLine(relative_rect=pygame.Rect((10, 70), (100, 30)), manager=self.manager, container=self.param_panel)
        self.e_input.set_text("0.3")
        self.e_label = pg.elements.UILabel(relative_rect=pygame.Rect((120, 70), (160, 30)), text="eccentricity", manager=self.manager, container=self.param_panel)

        self.inc_input = pg.elements.UITextEntryLine(relative_rect=pygame.Rect((10, 110), (100, 30)), manager=self.manager, container=self.param_panel)
        self.inc_input.set_text("15")
        self.inc_label = pg.elements.UILabel(relative_rect=pygame.Rect((120, 110), (160, 30)), text="inclination (°)", manager=self.manager, container=self.param_panel)

        self.mass_input = pg.elements.UITextEntryLine(relative_rect=pygame.Rect((10, 150), (100, 30)), manager=self.manager, container=self.param_panel)
        self.mass_input.set_text("400")
        self.mass_label = pg.elements.UILabel(relative_rect=pygame.Rect((120, 150), (160, 30)), text="mass (number of earth masses)", manager=self.manager, container=self.param_panel)

        self.doof_submit = pg.elements.UIButton(relative_rect=pygame.Rect((10, 190), (270, 30)), text="Create Doof's Planet", manager=self.manager, container=self.param_panel)

        self.prediction_label = pg.elements.UILabel(
    relative_rect=pygame.Rect((10, 230), (270, 20)), 
    text="Stability Prediction:", 
    manager=self.manager, 
    container=self.param_panel
)

        self.prediction_display = pg.elements.UITextBox(
    relative_rect=pygame.Rect((10, 255), (270, 90)),
    html_text="<font color='#AAAAAA'>No prediction yet...</font>",
    manager=self.manager,
    container=self.param_panel
)
        
        # Speed control slider
        self.speed_slider = pg.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((550, 780), (100, 50)),
            start_value=1.0, 
            value_range=(1, 1000.0),  
            manager=self.manager,
            click_increment=10  
        )
        
        self.speed_value_label = pg.elements.UILabel(relative_rect=pygame.Rect((650, 780), (140, 50)), text="1 year/sec", manager=self.manager)
        
       
        print(f"[log] Speed slider current value: {self.speed_slider.get_current_value()}")

        self._set_panel_visibility(False)
        self._set_param_panel_visibility(False)
        self._set_asteroid_panel_visibility(False)

    def _setup_intro_bubble(self):
       
      
        self.intro_bubble = pg.elements.UIPanel(
            relative_rect=pygame.Rect((850, 50), (400, 200)), 
            manager=self.manager
        )
        
        # Doof's message in the bubble
        bubble_text = """
        <font color='#FFD700'><b>I present to you, My Planet-Inator!</b></font><br>
        <font color='#CCCCCC'>Im the evil Scientist, Dr.Doofenshmirtz, I need to set up my HQ inanother planet  so that perrry the platypus wont be able to reach me </font><br>
        <font color='#CCCCCC'>And, creating another planet shouldnt be a problem, since our solar system has been stable for so long</font><br>
        <font color='#CCCCCC'> Right? </font><br><br>
        <font color='#FFFFFF'>Try it out!</font><br><br>
        <font color='#AAAAAA'>Press anywhere to dismiss</font>
        """
        
        self.intro_bubble_text = pg.elements.UITextBox(
            relative_rect=pygame.Rect((10, 10), (380, 180)),
            html_text=bubble_text,
            manager=self.manager,
            container=self.intro_bubble
        )
        
        
        self.intro_arrow = pg.elements.UILabel(
            relative_rect=pygame.Rect((300, 160), (80, 30)),
            text="👇 ",
            manager=self.manager,
            container=self.intro_bubble
        )
    
    def _setup_info_bubble(self, info_data, mouse_pos):
        """Create enhanced info bubble with rich asteroid data"""
        # Dynamic sizing based on content
        is_asteroid = info_data['type'] == 'asteroid'
        bubble_width = 450 if is_asteroid else 350
        bubble_height = 400 if is_asteroid else 250
        
        # Keep on screen
        x = min(mouse_pos[0], 1300 - bubble_width - 10)
        y = min(mouse_pos[1], 1000 - bubble_height - 10)
        
        if self.info_bubble:
            self.info_bubble.kill()
            self.info_bubble_text.kill()
        
        self.info_bubble = pg.elements.UIPanel(relative_rect=pygame.Rect(x, y, bubble_width, bubble_height), manager=self.manager)
        
        # Enhanced formatting
        html_text = f"<font color='#FFD700' size=5><b>{info_data['name']}</b></font><br>"
        html_text += f"<font color='#AAAAAA'>Type: {info_data['type'].title()}</font><br><br>"
        
        if is_asteroid:
            # Core orbital data
            if 'orbit_type' in info_data:
                html_text += f"<font color='#00FFFF'><b>Orbit Class:</b> {info_data['orbit_type']}</font><br>"
            
            if 'semi_major_axis' in info_data:
                html_text += f"<font color='#CCCCCC'><b>Distance:</b> {info_data['semi_major_axis']}</font><br>"
            
            if 'eccentricity' in info_data:
                html_text += f"<font color='#CCCCCC'><b>Eccentricity:</b> {info_data['eccentricity']}</font><br>"
            
            if 'inclination' in info_data:
                html_text += f"<font color='#CCCCCC'><b>Inclination:</b> {info_data['inclination']}</font><br>"
            
            # Physical properties
            if 'estimated_diameter' in info_data:
                html_text += f"<font color='#FFAA44'><b>Est. Diameter:</b> {info_data['estimated_diameter']}</font><br>"
            
            if 'absolute_magnitude' in info_data:
                html_text += f"<font color='#CCCCCC'><b>Brightness (H):</b> {info_data['absolute_magnitude']}</font><br>"
            
            # Hazard status
            if info_data.get('potentially_hazardous'):
                html_text += f"<font color='#FF4444'><b>⚠️ POTENTIALLY HAZARDOUS</b></font><br>"
            elif info_data.get('near_earth_object'):
                html_text += f"<font color='#FFAA00'><b>Near-Earth Object</b></font><br>"
            
            html_text += "<br>"
            
            # Detailed orbital info (collapsible sections)
            if 'synodic_period' in info_data:
                html_text += f"<font color='#AAFFAA'><b>Synodic Period:</b> {info_data['synodic_period']}</font><br>"
            
            if 'perihelion_distance' in info_data and 'aphelion_distance' in info_data:
                html_text += f"<font color='#CCCCCC'><b>Range:</b> {info_data['perihelion_distance']} - {info_data['aphelion_distance']}</font><br>"
            
            # Discovery info
            if 'catalog_number' in info_data:
                html_text += f"<font color='#AAAAAA'><b>Number:</b> {info_data['catalog_number']}</font><br>"
            
            if 'last_observation' in info_data:
                html_text += f"<font color='#AAAAAA'><b>Last Observed:</b> {info_data['last_observation']}</font><br>"
            
            if 'total_observations' in info_data:
                html_text += f"<font color='#AAAAAA'><b>Observations:</b> {info_data['total_observations']}</font><br>"
            
            # Orbit quality
            if 'orbit_uncertainty' in info_data:
                color = '#44FF44' if 'Very well' in info_data['orbit_uncertainty'] else '#FFAA00' if 'Well' in info_data['orbit_uncertainty'] else '#FF4444'
                html_text += f"<font color='{color}'><b>Orbit Quality:</b> {info_data['orbit_uncertainty']}</font><br>"
        
        else:  # Planet info
            if 'semi_major_axis' in info_data:
                html_text += f"<font color='#CCCCCC'><b>Distance:</b> {info_data['semi_major_axis']}</font><br>"
            if 'eccentricity' in info_data:
                html_text += f"<font color='#CCCCCC'><b>Eccentricity:</b> {info_data['eccentricity']}</font><br>"
            if 'inclination' in info_data:
                html_text += f"<font color='#CCCCCC'><b>Inclination:</b> {info_data['inclination']}</font><br>"
            if 'mass' in info_data:
                html_text += f"<font color='#CCCCCC'><b>Mass:</b> {info_data['mass']}</font><br>"
            if 'Orbital Period' in info_data:
                html_text += f"<font color='#CCCCCC'><b>Orbital Period:</b> {info_data['Orbital Period']}</font><br>"
        
        html_text += "<br><font color='#888888'>Click outside to close</font>"
        
        self.info_bubble_text = pg.elements.UITextBox(relative_rect=pygame.Rect(15, 15, bubble_width-30, bubble_height-30), html_text=html_text, manager=self.manager, container=self.info_bubble)
        
        self.info_bubble_visible = True
        self.info_bubble_data = info_data
    
    
    def _hide_info_bubble(self):
     """Hide the info bubble"""
     if self.info_bubble:
        self.info_bubble.kill()
        self.info_bubble_text.kill()
        self.info_bubble = None
        self.info_bubble_text = None
     self.info_bubble_visible = False
     self.info_bubble_data = None

    

    def handle_object_click(self, mouse_pos, world_pos):
    # larger half_size bigger tolerance
    # smaller half_sizesmaller tolerance
     base_planet_tolerance = 0.1
     base_asteroid_tolerance = 0.05
    

     zoom_scale_factor = max(1.0, self.viewport.half_size / 3.0)  # 3.0 is the initial viewport size
    
     planet_tolerance = base_planet_tolerance * zoom_scale_factor
     asteroid_tolerance = base_asteroid_tolerance * zoom_scale_factor
    
   
     for i, body in enumerate(self.sim.bodies):
        body_world_pos = (body.x - self.sim.bodies[0].x, body.y - self.sim.bodies[0].y)
        distance = ((world_pos[0] - body_world_pos[0])**2 + (world_pos[1] - body_world_pos[1])**2)**0.5
        
        if distance < planet_tolerance:
            info = self.sim.get_body_info(i)
            if info:
                self._setup_info_bubble(info, mouse_pos)
                return True
    
    
     for i, asteroid in enumerate(self.sim.asteroids):
        asteroid_world_pos = (asteroid.x - self.sim.bodies[0].x, asteroid.y - self.sim.bodies[0].y)
        distance = ((world_pos[0] - asteroid_world_pos[0])**2 + (world_pos[1] - asteroid_world_pos[1])**2)**0.5
        
        if distance < asteroid_tolerance:
            info = self.sim.get_asteroid_info(i)
            if info:
                self._setup_info_bubble(info, mouse_pos)
                return True
    
     return False


    def _hide_intro(self):
       
        self.show_intro = False
        self.intro_bubble.hide()
        self.intro_bubble_text.hide()
        self.intro_arrow.hide()
        
       
        self.dropdown_param_button.set_text(' Planet-inator')
    


    def _update_intro_timer(self):
        
        if self.show_intro:
            elapsed = time.time() - self.intro_start_time
            remaining = max(0, self.intro_duration - elapsed)
            
            # Fade in animation
            if self.bubble_alpha < self.bubble_target_alpha:
                self.bubble_alpha = min(self.bubble_target_alpha, self.bubble_alpha + 3)
            
            # Auto-hide after duration
            if remaining <= 0:
                self._hide_intro()
            else:
                # Add some animation to the arrow
                if int(elapsed * 3) % 2 == 0:  # Blink every 0.33 seconds
                    self.intro_arrow.set_text("👇 🌍")
                else:
                    self.intro_arrow.set_text("✨ 🌟")

    def _predict_stability_threaded(self, sim_copy):
        #Run SPOCK prediction in a separate thread
        try:
            print("[log] Starting SPOCK stability prediction...")
            
            deep_model = DeepRegressor()
            
            median, lower, upper, samples = deep_model.predict_instability_time(
                sim_copy, samples=5000, return_samples=True, seed=0
            )
            
            #  88 days is mercuries duration
            median_years = (median * 88) / 365.25
            lower_years = (lower * 88) / 365.25
            upper_years = (upper * 88) / 365.25
            
            # Put result in queue for main thread to pick up
            result = {
                'median': median_years,
                'lower': lower_years,
                'upper': upper_years,
                'success': True
            }
            self.prediction_queue.put(result)
            
            print(f"[log] SPOCK prediction complete: {median_years:.1f} years")
            
        except Exception as e:
            print(f"[log] SPOCK prediction failed: {e}")
            error_result = {
                'error': str(e),
                'success': False
            }
            self.prediction_queue.put(error_result)

    def _start_stability_prediction(self):
        """Start stability prediction in background thread"""
        if self.prediction_in_progress:
            return
            
        try:
            
            sim_copy = self.sim.sim.copy()
            sim_copy.move_to_com()
            
        
            self.prediction_display.set_text(
                "<font color='#FFAA00'>⚡ Doof's Calculating...</font>"
            )
            
            # Start prediction thread
            self.prediction_in_progress = True
            self.prediction_thread = threading.Thread(
                target=self._predict_stability_threaded, 
                args=(sim_copy,)
            )
            self.prediction_thread.daemon = True
            self.prediction_thread.start()
            
        except Exception as e:
            print(f"[log] Failed to start SPOCK prediction: {e}")
            self.prediction_display.set_text(
                f"<font color='#FF4444'>❌ Error: {str(e)}</font>"
            )

    def _check_prediction_result(self):
        """Check if prediction thread has completed"""
        if not self.prediction_queue.empty():
            result = self.prediction_queue.get()
            self.prediction_in_progress = False
            
            if result['success']:
                # Formatting
                median = result['median']
                lower = result['lower']
                upper = result['upper']
                
                if median < 70:
                    color = '#FF4444'  #
                    stability_text = " UNSTABLE!"
                elif median < 200:
                    color = '#FFAA00'  
                    stability_text = "RISKY"
                else:
                    color = '#44FF44'  
                    stability_text = "STABLE"
                
                prediction_text = f"""
                <font color='{color}'><b>{stability_text}</b></font><br>
                <font color='#CCCCCC'>Instability in:</font><br>
                <font color='#FFFFFF'>{median:.1f} years</font><br>
                <font color='#AAAAAA'>({lower:.1f} - {upper:.1f})</font>
                """
                
                self.prediction_display.set_text(prediction_text)
                
                
                '''if median < 1000:
                    self.doof_submit.set_text("Chaotic Planet! ")
                elif median < 10000:
                    self.doof_submit.set_text("Risky Planet! ")
                else:
                    self.doof_submit.set_text("Stable Planet! ")'''
                    
            else:
                self.prediction_display.set_text(
                    f"<font color='#FF4444'>Prediction failed:<br>{result['error']}</font>"
                )

    def _set_panel_visibility(self, visible):
        controls = [self.pause_button, self.zoom_in_button, self.zoom_out_button, 
                   self.reset_button, self.stars_button, self.screenshot_button, 
                   self.asteroid_options_button, self.quit_button]
        
        for btn in controls:
            btn.visible = visible
       
        if visible:
            self.panel.set_relative_position((10, 10))
            self.panel.set_dimensions((250, 230))
            self.dropdown_button.set_text('hide controls')
        else:
            self.panel.set_relative_position((10, 10))
            self.panel.set_dimensions((250, 20))
            self.dropdown_button.set_text('controls')

    def _set_asteroid_panel_visibility(self, visible):
        asteroid_controls = [self.asteroid_panel_title, self.nea_toggle_button, self.nea_orbits_button,
                           self.distant_toggle_button, self.distant_orbits_button, self.all_orbits_button]
        
        for element in asteroid_controls:
            element.visible = visible
        
        if visible:
            self.asteroid_panel.show()
            self.asteroid_options_button.set_text('▼ Asteroid Options')
        else:
            self.asteroid_panel.hide()
            self.asteroid_options_button.set_text('▶ Asteroid Options')

    def _set_param_panel_visibility(self, visible):
     for element in [self.a_input, self.a_label, self.e_input, self.e_label, self.inc_input, self.inc_label, self.mass_input, self.mass_label, self.doof_submit, self.prediction_label, self.prediction_display]:
        element.visible = visible
    
     if visible:
        self.param_panel.set_relative_position((1000, 430))
        self.param_panel.set_dimensions((290, 360))
        self.dropdown_param_button.set_text('hide')
     else:
        self.param_panel.set_relative_position((1000, 790))
        self.param_panel.set_dimensions((290, 20))
        # Reset button text if intro was skipped
        if not self.show_intro and not self.intro_clicked:
            self.dropdown_param_button.set_text('Planet-inator')
        elif self.intro_clicked:
            self.dropdown_param_button.set_text('Planet-inator')

    def draw_ui(self, screen):
       
        if self.show_intro and self.bubble_alpha > 0:
            bubble_surface = pygame.Surface((400, 200), pygame.SRCALPHA)
            bubble_color = (20, 20, 40, min(self.bubble_alpha, 200))  # Dark blue with transparency
            border_color = (100, 150, 255, min(self.bubble_alpha, 255))  # Light blue
            
         
            pygame.draw.rect(bubble_surface, bubble_color, (0, 0, 400, 200), border_radius=15)
            pygame.draw.rect(bubble_surface, border_color, (0, 0, 400, 200), width=2, border_radius=15)
            
           
            tail_points = [(350, 190), (370, 210), (330, 190)]
            pygame.draw.polygon(bubble_surface, bubble_color, tail_points)
            pygame.draw.polygon(bubble_surface, border_color, tail_points, width=2)
            
            # Blit the transparent bubble
            screen.blit(bubble_surface, (850, 50))
        
        # Draw the regular UI
        self.manager.draw_ui(screen)

    def process_events(self, event, screen, save_screenshot, output_video, video_writer):
        self.manager.process_events(event)
        
        # Handle intro screen events
        if self.show_intro and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._hide_intro()
        
       # click anywhere
        if self.show_intro and event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            bubble_rect = pygame.Rect(850, 50, 400, 200)
            if not bubble_rect.collidepoint(mouse_pos):
                self._hide_intro()
        
        if event.type == pygame.USEREVENT:
            if event.user_type == pg.UI_BUTTON_PRESSED:
                
                #button handling
                if event.ui_element == self.dropdown_button:
                    self.panel_collapsed = not self.panel_collapsed
                    self._set_panel_visibility(not self.panel_collapsed)
                elif event.ui_element == self.dropdown_param_button and not self.show_intro:
                    self.param_panel_collapsed = not self.param_panel_collapsed
                    self._set_param_panel_visibility(not self.param_panel_collapsed)
                elif event.ui_element == self.asteroid_options_button and not self.panel_collapsed:
                    self.asteroid_panel_collapsed = not self.asteroid_panel_collapsed
                    self._set_asteroid_panel_visibility(not self.asteroid_panel_collapsed)
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
                
                # Asteroid control buttons
                if not self.asteroid_panel_collapsed:
                    if event.ui_element == self.nea_toggle_button:
                        self.asteroid_visibility.nea_visible = not self.asteroid_visibility.nea_visible
                        print(f"Near Earth Asteroids {'visible' if self.asteroid_visibility.nea_visible else 'hidden'}")
                    elif event.ui_element == self.nea_orbits_button:
                        self.asteroid_visibility.nea_orbits_visible = not self.asteroid_visibility.nea_orbits_visible
                        print(f"NEA orbits {'visible' if self.asteroid_visibility.nea_orbits_visible else 'hidden'}")
                        self.viewport.zoom_changed = True  # Trigger redraw
                    elif event.ui_element == self.distant_toggle_button:
                        self.asteroid_visibility.distant_visible = not self.asteroid_visibility.distant_visible
                        print(f"Distant asteroids {'visible' if self.asteroid_visibility.distant_visible else 'hidden'}")
                    elif event.ui_element == self.distant_orbits_button:
                        self.asteroid_visibility.distant_orbits_visible = not self.asteroid_visibility.distant_orbits_visible
                        print(f"Distant asteroid orbits {'visible' if self.asteroid_visibility.distant_orbits_visible else 'hidden'}")
                        self.viewport.zoom_changed = True  # Trigger redraw
                    elif event.ui_element == self.all_orbits_button:
                        # This should trigger the same logic as the 'O' key
                        # need to access the asteroid_orbit_trail from the main file
                        print("Toggle all asteroid orbits - connect this to asteroid_orbit_trail.toggle_visibility()")
               
                if event.ui_element == self.doof_submit:
                    try:
                        print("[log] Planet-inator activated!")
                        
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
                        
                        # Update button text with evil flair
                        if self.sim.doof_planet_created:
                            self.doof_submit.set_text("🌍 Modify Evil Planet! 🌍")
                        else:
                            self.doof_submit.set_text("🌍 Planet Created! 🌍")
                        
                        # Start stability prediction
                        self._start_stability_prediction()
                        
                        print("[✓] Doof's diabolical planet has been created!")
                    except ValueError:
                        print("[!] Invalid input for Planet-inator!")
                        self.doof_submit.set_text("Invalid Parameters!")

            elif event.user_type == pg.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == self.speed_slider:
                    speed_multiplier = self.speed_slider.get_current_value()
                    base_delta_t = (14 * np.pi) / 100
                    self.sim.delta_t = base_delta_t * speed_multiplier
                    self.speed_value_label.set_text(f"{speed_multiplier:.1f} years/second")
                    print(f"[log] Speed set to {speed_multiplier:.1f}x (delta_t = {self.sim.delta_t:.4f})")

    def update(self, time_delta):
        self.manager.update(time_delta)
        
       # timer and animations
        self._update_intro_timer()
        
       # check
        self._check_prediction_result()
        
       
        current_slider_value = self.speed_slider.get_current_value()
        if abs(current_slider_value - self.last_slider_value) > 0.01:
            base_delta_t = (14 * np.pi) / 100
            self.sim.delta_t = base_delta_t * current_slider_value
            self.speed_value_label.set_text(f"{current_slider_value:.0f} years/second")
            self.last_slider_value = current_slider_value