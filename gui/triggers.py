import pygame
import pygame_gui as pg

class PygameGUIControls:
    def __init__(self, sim, viewport, paused_ref, stars_visible_ref):
        self.sim = sim
        self.viewport = viewport
        self.paused_ref = paused_ref  # Should be a dict or object with 'value' key/attr
        self.stars_visible_ref = stars_visible_ref  # Same as above
        self.manager = pg.UIManager((1000, 1000))
        self._setup_ui()

    def _setup_ui(self):
        # Panel
        #self.manager.set_theme(pg.themes.THEME_DARK)
        self.manager = pg.UIManager((800, 600), 'gui/theme.json')
        self.panel = pg.elements.UIPanel(relative_rect=pygame.Rect((10, 60), (250, 200)), manager=self.manager)
        # Buttons
        self.pause_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 10), (100, 30)), text='Pause', manager=self.manager, container=self.panel)
        self.zoom_in_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 50), (100, 30)), text='Zoom In', manager=self.manager, container=self.panel)
        self.zoom_out_button = pg.elements.UIButton(relative_rect=pygame.Rect((120, 50), (100, 30)), text='Zoom Out', manager=self.manager, container=self.panel)
        self.reset_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 90), (100, 30)), text='Reset', manager=self.manager, container=self.panel)
        self.stars_button = pg.elements.UIButton(relative_rect=pygame.Rect((120, 90), (100, 30)), text='Toggle Stars', manager=self.manager, container=self.panel)
        self.screenshot_button = pg.elements.UIButton(relative_rect=pygame.Rect((10, 130), (100, 30)), text='Screenshot', manager=self.manager, container=self.panel)
        self.quit_button = pg.elements.UIButton(relative_rect=pygame.Rect((120, 130), (100, 30)), text='Quit', manager=self.manager, container=self.panel)

    def draw_ui(self, screen):
        self.manager.draw_ui(screen)

    def process_events(self, event, screen, save_screenshot, output_video, video_writer):
        self.manager.process_events(event)
        if event.type == pygame.USEREVENT:
            if event.user_type == pg.UI_BUTTON_PRESSED:
                if event.ui_element == self.pause_button:
                    self.paused_ref['value'] = not self.paused_ref['value']
                    self.pause_button.set_text('Play' if self.paused_ref['value'] else 'Pause')
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