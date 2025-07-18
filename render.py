import time
from sim import Simulation
import pygame
import cv2 as cv
import click
import math
import numpy as np
import random
from gui.controllers import PygameGUIControls

WIDTH = 1300
HEIGHT = 850
INITIAL_VIEWPORT_HALF_SIZE = 3
MIN_ZOOM = 0.1  
MAX_ZOOM = 100 

SIZE = WIDTH, HEIGHT

NUM_STARS = 300  
STAR_MIN_BRIGHTNESS = 30
STAR_MAX_BRIGHTNESS = 180
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
        self.center_x = 0.0
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
        world_dx = (dx / WIDTH) * self.get_viewport_width()
        world_dy = -(dy / HEIGHT) * self.get_viewport_height()
        self.center_x -= world_dx
        self.center_y -= world_dy
        self.zoom_changed = True

class OrbitTrail:
    def __init__(self):
        self.trails = {}
    
    def add_position(self, body_id, x, y):
        if body_id not in self.trails:
            self.trails[body_id] = []
        
        self.trails[body_id].append((x, y))
    
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

def get_color_from_string(color_str):
    """Convert color string to RGB tuple"""
    color_map = {
        'red': (255, 0, 0),
        'orange': (255, 165, 0),
        'yellow': (255, 255, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'gray': (128, 128, 128),
        'white': (255, 255, 255),
        'lightblue': (173, 216, 230),
        'goldenrod': (218, 165, 32),
        'black': (0, 0, 0)
    }
    return color_map.get(color_str.lower(), (255, 255, 255))

def is_asteroid(body):
    """Check if a body is an asteroid based on its name and size"""
    # Check if it's a planet or sun (these are not asteroids)
    major_bodies = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 
                   'saturn', 'uranus', 'neptune', 'doofs-planet']
    
    if body.name.lower() in major_bodies:
        return False
    
    # If it's very small, it's likely an asteroid
    if body.size < 0.1:
        return True
    
    # Check for common asteroid indicators in the name
    asteroid_indicators = ['asteroid', 'eros', 'bennu', 'apophis', 'itokawa', 'ryugu',
                          'vesta', 'ceres', 'pallas', 'juno', 'toutatis', 'gaspra']
    
    # Check if name contains numbers (many asteroids have numerical designations)
    if any(char.isdigit() for char in body.name):
        return True
    
    return any(indicator in body.name.lower() for indicator in asteroid_indicators)

def is_interesting_asteroid(body):
    """Check if an asteroid is interesting enough to show labels"""
    interesting_names = ['eros', 'bennu', 'apophis', 'itokawa', 'ryugu', 'vesta', 
                        'ceres', 'pallas', 'juno', 'toutatis', 'gaspra']
    return any(name in body.name.lower() for name in interesting_names)

def count_asteroids(bodies):
    """Count how many asteroids are in the simulation"""
    return sum(1 for body in bodies if is_asteroid(body))

def get_body_display_size(body, viewport):
    """Calculate appropriate display size for a body"""
    if is_asteroid(body):
        # For asteroids, use a minimum size that's visible but scale with zoom
        base_size = max(10, body.size * 200000)  # Scale up asteroid size
        zoom_scale = max(0.5, viewport.zoom_factor * 0.3)  # Scale with zoom
        return int(base_size * zoom_scale)
    else:
        # For planets and sun, use original scaling
        return max(int((body.size * WIDTH) / (viewport.half_size * 2)), 3)

@click.command()
@click.option('--output_video', '-v', default='', help='Filename of video to write to')
@click.option('--load_nea', '-n', default='', help='Path to NEA JSON file')
@click.option('--nea_count', '-c', default=50, help='Number of NEAs to load')
def main(output_video, load_nea, nea_count):
    sim = Simulation()
    
    # Load NEA data if provided
    if load_nea:
        try:
            loaded_count = sim.load_nea_data(load_nea, limit=1000)
            print(f"Loaded {loaded_count} NEAs from {load_nea}")
            
            # Add interesting asteroids
            sim.add_asteroids(max_count=nea_count, interesting_only=True)
            print(f"Added {nea_count} asteroids to simulation")
        except Exception as e:
            print(f"Error loading NEAs: {e}")
            print("Continuing without NEAs...")

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
    show_asteroids = {'value': True}
    show_asteroid_trails = {'value': False}

    # Use GUI controls
    gui_controls = PygameGUIControls(sim, viewport, paused, stars_visible)

    screen.fill((0, 0, 0))
    
    video_writer = None
    if output_video:
        pygame.display.update()
        video_writer = cv.VideoWriter(output_video, cv.VideoWriter_fourcc(*'DIVX'), 30, screen.get_size())

    delay = 0.025
    log_t = 0
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
            asteroid_count_rendered = 0
            for i, body in enumerate(sim.bodies):
                
                trail_positions = orbit_trail.get_trail(i)
                body_color = get_color_from_string(body.color)
                
                # Skip asteroid trails if disabled
                if not show_asteroid_trails['value'] and is_asteroid(body):
                    continue
                    
                for trail_x, trail_y in trail_positions:
                    trail_pixel_pos = viewport_to_pixels(trail_x, trail_y)
                    if trail_pixel_pos:
                        pygame.draw.circle(orbits_surface, body_color, trail_pixel_pos, 1)
               
            viewport.zoom_changed = False
            print(f"Rendered {asteroid_count_rendered} asteroids this frame")

        if sim.t == 0:
            orbits_surface.fill((0, 0, 0, 0))
            orbit_trail.trails.clear()
        
        # Add current positions to orbit trails and draw new trail points
        for i, body in enumerate(sim.bodies):
            current_pos = (body.x - sim.bodies[0].x, body.y - sim.bodies[0].y)
            
            # Add trails for all bodies if enabled
            if show_asteroid_trails['value'] or not is_asteroid(body):
                # Only add and draw if this is a new position
                trail_positions = orbit_trail.get_trail(i)
                if not trail_positions or trail_positions[-1] != current_pos:
                    orbit_trail.add_position(i, current_pos[0], current_pos[1])
                    
                    # Draw only the new trail point
                    trail_pixel_pos = viewport_to_pixels(current_pos[0], current_pos[1])
                    if trail_pixel_pos:
                        body_color = get_color_from_string(body.color)
                        pygame.draw.circle(orbits_surface, body_color, trail_pixel_pos, 1)

        # Render celestial bodies
        for i, body in enumerate(sim.bodies):
            position = viewport_to_pixels(body.x - sim.bodies[0].x, body.y - sim.bodies[0].y)
            if position:
                # Skip asteroids if disabled
                if is_asteroid(body) and not show_asteroids['value']:
                    continue
                
                # Calculate size based on body type
                size = get_body_display_size(body, viewport)
                
                # Sun glow effect
                if i == 0:
                    glow_color = (255, 240, 180)
                    max_glow_radius = size * 4
                    steps = 100
                    for step in range(steps):
                        r = int(size + (max_glow_radius - size) * (step / steps))
                        alpha = int(3 * (1 - (step / steps)))
                        if alpha <= 0:
                            continue
                        glow_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                        pygame.draw.circle(glow_surf, (*glow_color, alpha), (int(position[0]), int(position[1])), r)
                        bodies_surface.blit(glow_surf, (0, 0))
                
                # Draw the body
                body_color = get_color_from_string(body.color)
                pygame.draw.circle(bodies_surface, body_color, position, size)
                
                # Draw name labels for interesting bodies when zoomed in
                if viewport.zoom_factor > 2:
                    # Show labels for planets and interesting asteroids
                    if not is_asteroid(body) or is_interesting_asteroid(body):
                        name_text = font.render(body.name, True, (255, 255, 255))
                        name_pos = (position[0] + size + 5, position[1] - 10)
                        bodies_surface.blit(name_text, name_pos)
                
                # Show asteroid count when zoomed out
                elif viewport.zoom_factor < 1 and is_asteroid(body):
                    # When zoomed out, show a small dot for asteroids
                    pygame.draw.circle(bodies_surface, body_color, position, max(1, size//2))
            
        screen.blit(orbits_surface, (0, 0))
        screen.blit(bodies_surface, (0, 0))

        # Draw UI
        gui_controls.draw_ui(screen)

        # Display information
        zoom_text = font.render(f"Zoom: {viewport.zoom_factor:.2f}x", True, (255, 255, 255))
        years_text = font1.render(f"Years passed: {get_time(sim.t):.1f}", True, (255, 255, 255))
        
        # Display asteroid count
        asteroid_count = count_asteroids(sim.bodies)
        asteroid_text = font.render(f"Asteroids: {asteroid_count}", True, (255, 255, 255))
        
        # Display total bodies
        total_bodies_text = font.render(f"Total bodies: {len(sim.bodies)}", True, (255, 255, 255))
        
        screen.blit(zoom_text, (1200, 10))
        screen.blit(years_text, (10, 800))
        screen.blit(asteroid_text, (10, 780))
        screen.blit(total_bodies_text, (10, 760))

        # Toggle instructions
        instructions = [
            "T - Toggle stars",
            "A - Toggle asteroids", 
            "R - Toggle asteroid trails",
            "P - Pause/Resume",
            "S - Screenshot",
            "Mouse wheel - Zoom",
            "Drag - Pan"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_text = font.render(instruction, True, (200, 200, 200))
            screen.blit(inst_text, (10, 10 + i * 15))

        pygame.display.update()

        if output_video:
            video_writer.write(image_from_screen(screen))

        if not paused['value']:
            sim.iterate()
            t_delta = time.time() - t_start
            if t_delta < delay:
                time.sleep(delay - t_delta)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if output_video:
                    video_writer.release()
                pygame.quit()
                quit()
            
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
                elif event.key == pygame.K_a:  # Toggle asteroids
                    show_asteroids['value'] = not show_asteroids['value']
                    print(f"Asteroids: {'ON' if show_asteroids['value'] else 'OFF'}")
                elif event.key == pygame.K_r:  # Toggle asteroid trails
                    show_asteroid_trails['value'] = not show_asteroid_trails['value']
                    print(f"Asteroid trails: {'ON' if show_asteroid_trails['value'] else 'OFF'}")
                elif event.key == pygame.K_PLUS:
                    viewport.zoom_in()
                elif event.key == pygame.K_MINUS:
                    viewport.zoom_out()
                elif event.key == pygame.K_EQUALS:
                    viewport.reset_zoom()
                    print(f"Zoom reset: {viewport.zoom_factor:.2f}x")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up
                    viewport.zoom_in()
                elif event.button == 5:  # Mouse wheel down
                    viewport.zoom_out()
                elif event.button == 1:  # Left mouse button
                    dragging = True
                    drag_start_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
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