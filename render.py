
import time
from rebound.sim import Simulation
import pygame
import cv2 as cv
import click
import math
import numpy as np
import random
from gui.controllers import PygameGUIControls




WIDTH = 1300
#WIDTH = 1450
HEIGHT = 850
INITIAL_VIEWPORT_HALF_SIZE = 3 #change this for closer look at inner planets (=3) 
MIN_ZOOM = 0.1  
MAX_ZOOM = 100 

SIZE = WIDTH, HEIGHT

NUM_STARS = 300  
STAR_MIN_BRIGHTNESS = 30  # 
STAR_MAX_BRIGHTNESS = 180  # out of 255
STAR_MIN_SIZE = 1
STAR_MAX_SIZE = 1.5

# Generate star field once
star_field = []
for _ in range(NUM_STARS):
    x = random.randint(0, WIDTH-1)
    y = random.randint(0, HEIGHT-1)
    brightness = random.randint(STAR_MIN_BRIGHTNESS, STAR_MAX_BRIGHTNESS)
    size = random.uniform(STAR_MIN_SIZE, STAR_MAX_SIZE)
    star_field.append((x, y, brightness, size))

class Zwoom:
    def __init__(self, initial_half_size):
        self.half_size = initial_half_size
        self.zoom_factor = 1.0
        self.zoom_changed = False
        self.center_x = 0.0  # Viewport center offset
        self.center_y = 0.0
    
    def get_viewport(self):
        return ((self.center_x - self.half_size, self.center_y - self.half_size), 
                (self.center_x + self.half_size, self.center_y + self.half_size))
    
    def get_viewport_width(self):
        return self.half_size * 2
    
    def get_viewport_height(self):
        return self.half_size * 2
    
    def zoom_in(self, factor=1.2):
        new_half_size = self.half_size / factor
        if new_half_size >= MIN_ZOOM:
            self.half_size = new_half_size
            self.zoom_factor *= factor
            self.zoom_changed = True
    
    def zoom_out(self, factor=1.2):
        new_half_size = self.half_size * factor
        if new_half_size <= MAX_ZOOM:
            self.half_size = new_half_size
            self.zoom_factor /= factor
            self.zoom_changed = True
    
    def reset_zoom(self):
        self.half_size = INITIAL_VIEWPORT_HALF_SIZE
        self.zoom_factor = 1.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.zoom_changed = True
    
    def pan(self, dx, dy):
        # drag - world cordinates
        world_dx = (dx / WIDTH) * self.get_viewport_width()
        world_dy = -(dy / HEIGHT) * self.get_viewport_height()  # Invert Y
        self.center_x -= world_dx
        self.center_y -= world_dy
        self.zoom_changed = True

class OrbitTrail:
    def __init__(self):
        self.trails = {}  # body_id -> (x, y)
    
    def add_position(self, body_id, x, y):
        if body_id not in self.trails:
            self.trails[body_id] = []
        
        self.trails[body_id].append((x, y))
        # no fade
    
    def get_trail(self, body_id):
        return self.trails.get(body_id, [])

# Global viewport instance
viewport = Zwoom(INITIAL_VIEWPORT_HALF_SIZE)
orbit_trail = OrbitTrail()
def viewport_to_pixels(x, y):
    current_viewport = viewport.get_viewport()
    viewport_width = viewport.get_viewport_width()
    viewport_height = viewport.get_viewport_height()
    
    if x < current_viewport[0][0] or x > current_viewport[1][0]: 
        return None
    if y < current_viewport[0][1] or y > current_viewport[1][1]:
        return None

    px = ((x - current_viewport[0][0]) * WIDTH) / viewport_width
    py = ((viewport_height - (y - current_viewport[0][1])) * HEIGHT) / viewport_height

    return px, py

@click.command()
@click.option('--output_video', '-v', default='', help='Filename of video to write to')
def main(output_video):
    sim = Simulation()

    pygame.init()

    screen = pygame.display.set_mode(SIZE)
    orbits_surface = pygame.Surface(SIZE, pygame.SRCALPHA)
    bodies_surface = pygame.Surface(SIZE, pygame.SRCALPHA)
    bodies_surface.set_colorkey((0, 0, 0))
    font = pygame.font.Font('freesansbold.ttf', 13)
    font1 = pygame.font.Font('freesansbold.ttf', 20)

    # State references for GUI
    paused = {'value': False}
    stars_visible = {'value': True}

    # --- Use new GUI controls ---
    gui_controls = PygameGUIControls(sim, viewport, paused, stars_visible)

    screen.fill((0, 0, 0))
    
    video_writer = None  # Ensure video_writer is always defined
    if output_video:
        pygame.display.update()  # force display to assume actual size
        video_writer = cv.VideoWriter(output_video, cv.VideoWriter_fourcc(*'DIVX'), 30, screen.get_size())

    delay = 0.025 # 25ms bw frames
    log_t = 0 # time tracker for logging
    month = 11
    week = 44
    
    dragging = False
    drag_start_pos = None 

    clock = pygame.time.Clock()

    while True:
        time_delta = clock.tick(60) / 1000.0
        t_start = time.time()

        # Draw star field background if enabled
        screen.fill((0, 0, 0))
        if stars_visible['value']:
            for x, y, brightness, size in star_field:
                color = (brightness, brightness, brightness)
                pygame.draw.circle(screen, color, (x, y), int(size))

        bodies_surface.fill((0, 0, 0, 0))

        # Clear orbit surface only when zoom changes
        if viewport.zoom_changed:
            orbits_surface.fill((0, 0, 0, 0))
            # Redraw all existing trail points at new zoom level
            for i, body in enumerate(sim.bodies):
                trail_positions = orbit_trail.get_trail(i)
                for trail_x, trail_y in trail_positions:
                    trail_pixel_pos = viewport_to_pixels(trail_x, trail_y)
                    if trail_pixel_pos:
                        pygame.draw.circle(orbits_surface, body.color, trail_pixel_pos, 1)
            viewport.zoom_changed = False

        if sim.t == 0:
            orbits_surface.fill((0, 0, 0, 0))
            orbit_trail.trails.clear()
        
        # Add current positions to orbit trails and draw new trail points
        for i, body in enumerate(sim.bodies):
            current_pos = (body.x - sim.bodies[0].x, body.y - sim.bodies[0].y)
            
            # Only add and draw if this is a new position
            trail_positions = orbit_trail.get_trail(i)
            if not trail_positions or trail_positions[-1] != current_pos:
                orbit_trail.add_position(i, current_pos[0], current_pos[1])
                
                # Draw only the new trail point
                trail_pixel_pos = viewport_to_pixels(current_pos[0], current_pos[1])
                if trail_pixel_pos:
                    pygame.draw.circle(orbits_surface, body.color, trail_pixel_pos, 1)


    #time to render the celestial balls :p
        for i, body in enumerate(sim.bodies):
         position = viewport_to_pixels(body.x - sim.bodies[0].x, body.y - sim.bodies[0].y) #distance relative to mc (sun)
         if position:
                size = max((body.size * WIDTH) / (viewport.half_size * 2), 5) #change this
                #sun gelow
                if i == 0:
                    glow_color = (255, 240, 180)
                    max_glow_radius = size * 4
                    steps = 100
                    for step in range(steps):
                        r = int(size + (max_glow_radius - size) * (step / steps))
                        alpha = int(3 * (1 - (step / steps)))  # gatekeeping this
                        if alpha <= 0:
                            continue
                        glow_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                        pygame.draw.circle(glow_surf, (*glow_color, alpha), (int(position[0]), int(position[1])), r)
                        bodies_surface.blit(glow_surf, (0, 0))
                pygame.draw.circle(bodies_surface, body.color, position, size)
            
        screen.blit(orbits_surface, (0, 0))  #dahes 
        screen.blit(bodies_surface, (0, 0))

        # 
        gui_controls.draw_ui(screen)

        zoom_text = font.render(f"Zoom: {viewport.zoom_factor:.2f}x \nCenter: {viewport.center_x:.2f}, {viewport.center_y:.2f} \nHalf-size: {viewport.half_size:.2f} \n", True, (255, 255, 255))
        years_text = font1.render(f"no of years passed: {get_time(sim.t):.0f}", True, (255, 255, 255))
        screen.blit(zoom_text, (1200, 10))
        screen.blit(years_text, (10, 800))

        sun_distance = calculate_dist(sim.bodies[0], sim.bodies[3])

        #logging my beloved
        '''if get_time(sim.t) > log_t: 
            print(f't={get_time(sim.t):.2f}, month {int((week-1)/4.3)+1:2d}, week {week:2d}: {sun_distance}')
            log_t += 1/52
            if week == 52:
                week = 1
            else:
                week += 1'''

        pygame.display.update()

        if output_video:
            video_writer.write(image_from_screen(screen))

        if not paused['value']:
            sim.iterate() #sim step one at a time
            t_delta = time.time() - t_start    #frame rate control
            if t_delta < delay:
                time.sleep(delay - t_delta)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if output_video:
                    video_writer.release()
                pygame.quit()
                quit()
            # gui starts here (hell)
            gui_controls.process_events(event, screen, save_screenshot, output_video, video_writer)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused['value'] = not paused['value']
                elif event.key == pygame.K_s:
                    save_screenshot(screen, sim.t)
                elif event.key == pygame.K_q:
                    if output_video:
                        video_writer.release()
                    pygame.quit()
                    quit()
                elif event.key == pygame.K_t:
                    stars_visible['value'] = not stars_visible['value']
                elif event.key == pygame.K_PLUS:
                    viewport.zoom_in()
                elif event.key == pygame.K_MINUS:
                    viewport.zoom_out()
                elif event.key == pygame.K_r or event.key == pygame.K_EQUALS:
                    viewport.reset_zoom()
                    print(f"Zoom reset: {viewport.zoom_factor:.2f}x")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:  #fingers
                if event.button == 4:  # Mouse wheel up # what
                    viewport.zoom_in()
                elif event.button == 5:  # Mouse wheel down
                    viewport.zoom_out()
                elif event.button == 1:  # Left mouse button
                    dragging = True
                    drag_start_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    dragging = False
                    drag_start_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if dragging and drag_start_pos:
                    current_pos = pygame.mouse.get_pos()
                    dx = current_pos[0] - drag_start_pos[0]
                    dy = current_pos[1] - drag_start_pos[1]
                    viewport.pan(dx, dy)
                    drag_start_pos = current_pos
        gui_controls.update(time_delta)

def save_screenshot(screen, t):
    image = image_from_screen(screen)
    t_str = format_time(t).replace('.', '_')
    filename = f'moments/year_t_{t_str}.png'
    cv.imwrite(filename, image)
    print(f'Wrote screenshot {filename}')
    pass


def image_from_screen(screen):
    assert screen.get_bytesize() == 4, "Screen should be RGBA"
    width, height = screen.get_size()
    buffer = screen.get_buffer().raw
    image = np.ndarray((width, height, screen.get_bytesize()), 'uint8', buffer)
    return cv.cvtColor(image, cv.COLOR_BGRA2BGR)


def format_time(t):
    return str(round(t/(2*np.pi), 1))


def get_time(t):
    return t/(2*np.pi)


def calculate_dist(body1, body2):
    a = body1.x - body2.x
    b = body1.y - body2.y
    return math.sqrt((a*a) + (b*b))


if __name__ == '__main__':
    main()









