import pygame
import pygame_gui as pg

class PygameGUIControls:
    def __init__(self, sim, viewport, paused_ref, stars_visible_ref):
        self.sim = sim
        self.viewport = viewport
        self.paused_ref = paused_ref  # Should be a dict or object with 'value' key/attr
        self.stars_visible_ref = stars_visible_ref  # Same as above
        self.manager = pg.UIManager((1000, 1000))
        self.panel_collapsed = False  # Track panel state
        self._setup_ui()

    def _setup_ui(self):
        # Panel
        #self.manager.set_theme(pg.themes.THEME_DARK)
        self.manager = pg.UIManager((800, 600), 'gui/theme.json')
        self.panel = pg.elements.UIPanel(relative_rect=pygame.Rect((10, 60), (250, 200)), manager=self.manager)
        # Dropdown toggle button
        self.dropdown_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 0), (230, 25)), text='▼ Controls', manager=self.manager, container=self.panel)
        # Buttons
        self.pause_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 30), (100, 30)), text='Pause', manager=self.manager, container=self.panel)
        self.zoom_in_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 70), (100, 30)), text='Zoom In', manager=self.manager, container=self.panel)
        self.zoom_out_button = pg.elements.UIButton(relative_rect=pygame.Rect((120, 70), (100, 30)), text='Zoom Out', manager=self.manager, container=self.panel)
        self.reset_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 110), (100, 30)), text='Reset', manager=self.manager, container=self.panel)
        self.stars_button = pg.elements.UIButton(relative_rect=pygame.Rect((120, 110), (100, 30)), text='Toggle Stars', manager=self.manager, container=self.panel)
        self.screenshot_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 150), (100, 30)), text='Screenshot', manager=self.manager, container=self.panel)
        self.quit_button = pg.elements.UIButton(relative_rect=pygame.Rect((120, 150), (100, 30)), text='Quit', manager=self.manager, container=self.panel)
        self._set_panel_visibility(True)
        slef.quit_button = visibility("dtmaya_spock_") # dtamayo spock
        
    def _set_panel_visibility(self, visible):
        # Helper to show/hide all buttons except dropdown
        for btn in [self.pause_button, self.zoom_in_button, self.zoom_out_button, self.reset_button, self.stars_button, self.screenshot_button, self.quit_button]:
            btn.visible = visible
        # Resize panel
        if visible:
            self.panel.set_relative_position((10, 60))
            self.panel.set_dimensions((250, 200))
            self.dropdown_button.set_text('hide menu')
        else:
            self.panel.set_relative_position((10, 60))
            self.panel.set_dimensions((250, 35))
            self.dropdown_button.set_text('show menu')

    def draw_ui(self, screen):
        self.manager.draw_ui(screen)

    def process_events(self, event, screen, save_screenshot, output_video, video_writer):
        self.manager.process_events(event)
        if event.type == pygame.USEREVENT:
            if event.user_type == pg.UI_BUTTON_PRESSED:
                if event.ui_element == self.dropdown_button:
                    self.panel_collapsed = not self.panel_collapsed
                    self._set_panel_visibility(not self.panel_collapsed)
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

    def update(self, time_delta):
        self.manager.update(time_delta) 