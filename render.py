import time
from sim import Simulation
import pygame
import cv2 as cv
import click
import math
import numpy as np
import random
from gui.controllers import PygameGUIControls
from webcolors import name_to_rgb




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
    
planet_textures = {}

def load_planet_textures():
    texture_files = {
        'Sun': 'textures/2k_sun.jpg',
        'Mercury': 'textures/2k_mercury.jpg',
        'Venus': 'textures/2k_venus.jpg',
        'Earth': 'textures/2k_earth.jpg',
        'Mars': 'textures/2k_mars.jpg',
        'Jupiter': 'textures/2k_jupiter.jpg',
        'Saturn': 'textures/2k_saturn.jpg',
        'Uranus': 'textures/2k_uranus.jpg',
        'Neptune': 'textures/2k_neptune.jpg'
    }
    
    for name, file in texture_files.items():
        try:
            planet_textures[name] = pygame.image.load(file)
        except pygame.error:
            planet_textures[name] = None

def draw_planet(surface, body, position, size, t):
    texture = planet_textures.get(body.name)
    if texture and size > 6:
        # Create a circular surface
        circle_size = int(size * 2)
        circle_surface = pygame.Surface((circle_size, circle_size), pygame.SRCALPHA)
        
        # Scale and rotate texture
        scaled_texture = pygame.transform.scale(texture, (circle_size, circle_size))
        if body.name != 'Sun':
            rotation = (t * 50 * (1.0 / max(0.1, getattr(body.particle, 'a', 1.0)))) % 360
            scaled_texture = pygame.transform.rotate(scaled_texture, rotation)
            # Re-center after rotation
            temp_rect = scaled_texture.get_rect()
            circle_surface = pygame.Surface((temp_rect.width, temp_rect.height), pygame.SRCALPHA)
        
        # Draw texture onto circular surface
        circle_surface.blit(scaled_texture, (0, 0))
        
        # Clip to circle using a temporary surface
        temp_surface = pygame.Surface((circle_size, circle_size), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, (255, 255, 255, 255), (circle_size//2, circle_size//2), int(size))
        
        # Use the circle as an alpha mask
        final_surface = pygame.Surface((circle_size, circle_size), pygame.SRCALPHA)
        final_surface.blit(scaled_texture, (0, 0))
        final_surface.blit(temp_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        rect = final_surface.get_rect(center=position)
        surface.blit(final_surface, rect)
    else:
        pygame.draw.circle(surface, body.color, position, int(size))
    
class AsteroidTrail:
    def __init__(self):
        self.trails = {}  # body_id -> (x, y)
        self.visible = True  # Add visibility toggle
    
    def add_position(self, body_id, x, y):
        if body_id not in self.trails:
            self.trails[body_id] = []
        
        self.trails[body_id].append((x, y))
        # no fade
    
    def get_trail(self, body_id):
        return self.trails.get(body_id, [])
    
    def toggle_visibility(self):
        self.visible = not self.visible

# Asteroid visibility state
class AsteroidVisibility:
    def __init__(self):
        self.nea_visible = True
        self.nea_orbits_visible = True
        self.distant_visible = True
        self.distant_orbits_visible = True

# Global viewport instance
viewport = Zwoom(INITIAL_VIEWPORT_HALF_SIZE)
orbit_trail = OrbitTrail()
asteroid_orbit_trail = AsteroidTrail()
asteroid_visibility = AsteroidVisibility()

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
 
def pixels_to_viewport(px, py):
   
    current_viewport = viewport.get_viewport()
    viewport_width = viewport.get_viewport_width()
    viewport_height = viewport.get_viewport_height()
    
    world_x = current_viewport[0][0] + (px * viewport_width) / WIDTH
    world_y = current_viewport[0][1] + ((HEIGHT - py) * viewport_height) / HEIGHT
    
    return world_x, world_y

def should_draw_asteroid(k, asteroid, sim):
    nea_count = getattr(sim, 'nea_count', 0)
    
    if k < nea_count:  # NEA asteroid
        return asteroid_visibility.nea_visible
    else:  # Distant asteroid
        return asteroid_visibility.distant_visible

def should_draw_asteroid_orbit(k, asteroid, sim):
    nea_count = getattr(sim, 'nea_count', 0)
    
    if k < nea_count:  # NEA asteroid
        return asteroid_visibility.nea_orbits_visible
    else:  # Distant asteroid
        return asteroid_visibility.distant_orbits_visible

@click.command()
@click.option('--output_video', '-v', default='', help='Filename of video to write to')
@click.option('--nea_asteroids', '-n', default=50, help='Number of NEAs to load')
@click.option('--distant_asteroids', '-d', default=65, help='Number of distant asteroids to load')
def main(output_video, nea_asteroids, distant_asteroids):

    sim = Simulation()
    
    # Load asteroids if requested and store counts
    if nea_asteroids > 0 or distant_asteroids > 0:
        print(f"Loading {nea_asteroids} NEA and {distant_asteroids} distant asteroids...")
        sim.load_asteroids(nea_count=nea_asteroids, distant_count=distant_asteroids)
        # Store counts for visibility checking
        sim.nea_count = nea_asteroids
        sim.distant_count = distant_asteroids


    pygame.init()
    load_planet_textures()

    screen = pygame.display.set_mode(SIZE)
    asteroid_orbits_surface = pygame.Surface(SIZE, pygame.SRCALPHA)
    orbits_surface = pygame.Surface(SIZE, pygame.SRCALPHA)
    bodies_surface = pygame.Surface(SIZE, pygame.SRCALPHA)
    bodies_surface.set_colorkey((0, 0, 0))
    font = pygame.font.Font('freesansbold.ttf', 13)
    font1 = pygame.font.Font('freesansbold.ttf', 20)
    font_small = pygame.font.Font('freesansbold.ttf', 10)

    # State references for GUI
    paused = {'value': False}
    stars_visible = {'value': True}
  

    
    gui_controls = PygameGUIControls(sim, viewport, paused, stars_visible, asteroid_visibility)

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
            asteroid_orbits_surface.fill((0,0,0,0))
            # Redraw all existing trail points at new zoom level
            if asteroid_orbit_trail.visible:  # Only redraw if visible
                for k, asteroid in enumerate(sim.asteroids):
                    if should_draw_asteroid_orbit(k, asteroid, sim):
                        trail_positions = asteroid_orbit_trail.get_trail(k)
                        #lighter_color = lighten_color(asteroid.color, 0.4)  # Make asteroid orbits lighter
                        for trail_x, trail_y in trail_positions:
                            trail_pixel_pos = viewport_to_pixels(trail_x, trail_y)
                            if trail_pixel_pos:
                                pygame.draw.circle( asteroid_orbits_surface, asteroid.color, trail_pixel_pos, 1)
            for i, body in enumerate(sim.bodies):
                trail_positions = orbit_trail.get_trail(i)
                #lighter_color = lighten_color(body.color, 0.4)  # Make planet orbits lighter
                for trail_x, trail_y in trail_positions:
                    trail_pixel_pos = viewport_to_pixels(trail_x, trail_y)
                    if trail_pixel_pos:
                        pygame.draw.circle(orbits_surface, body.color, trail_pixel_pos, 1)

            viewport.zoom_changed = False

        if sim.t == 0:
            orbits_surface.fill((0, 0, 0, 0))
            orbit_trail.trails.clear()
            asteroid_orbits_surface.fill((0, 0, 0, 0))
            asteroid_orbit_trail.trails.clear()
        
        # Only update asteroid trails if they're visible
        if asteroid_orbit_trail.visible:
            for k, asteroid in enumerate(sim.asteroids):
                current_pos = (asteroid.x - sim.bodies[0].x, asteroid.y - sim.bodies[0].y)
                
                # Only add and draw if this is a new position and should be visible
                trail_positions = asteroid_orbit_trail.get_trail(k)
                if not trail_positions or trail_positions[-1] != current_pos:
                    asteroid_orbit_trail.add_position(k, current_pos[0], current_pos[1])
                    
                    # Draw only the new trail point with lighter color if orbit should be visible
                    if should_draw_asteroid_orbit(k, asteroid, sim):
                        trail_pixel_pos = viewport_to_pixels(current_pos[0], current_pos[1])
                        if trail_pixel_pos:
                           # lighter_color = lighten_color(asteroid.color, 0.2)  # Make asteroid orbits lighter
                            pygame.draw.circle(asteroid_orbits_surface, asteroid.color, trail_pixel_pos, 1)
        else:
            # Still add positions to trails even when not visible, just don't draw them
            for k, asteroid in enumerate(sim.asteroids):
                current_pos = (asteroid.x - sim.bodies[0].x, asteroid.y - sim.bodies[0].y)
                trail_positions = asteroid_orbit_trail.get_trail(k)
                if not trail_positions or trail_positions[-1] != current_pos:
                    asteroid_orbit_trail.add_position(k, current_pos[0], current_pos[1])
        
        # Add current positions to orbit trails and draw new trail points
        for i, body in enumerate(sim.bodies):
            current_pos = (body.x - sim.bodies[0].x, body.y - sim.bodies[0].y)
            
            # Only add and draw if this is a new position
            trail_positions = orbit_trail.get_trail(i)
            if not trail_positions or trail_positions[-1] != current_pos:
                orbit_trail.add_position(i, current_pos[0], current_pos[1])
                
                # Draw only the new trail point with lighter color
                trail_pixel_pos = viewport_to_pixels(current_pos[0], current_pos[1])
                if trail_pixel_pos:
                  #  lighter_color = lighten_color(body.color, 0.4)  # Make planet orbits lighter
                    pygame.draw.circle(orbits_surface, body.color, trail_pixel_pos, 1)



    #time to render the celestial balls :p
        for i, body in enumerate(sim.bodies):
         position = viewport_to_pixels(body.x - sim.bodies[0].x, body.y - sim.bodies[0].y) #distance relative to mc (sun)
         if position:
                size = max((body.size * WIDTH) / (viewport.half_size * 2), 3) #change this - minimum size 3 for asteroids
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
                draw_planet(bodies_surface, body, position, size, sim.t)

        for k, asteroid in enumerate(sim.asteroids):
            if should_draw_asteroid(k, asteroid, sim):  # Check if asteroid should be visible
                position = viewport_to_pixels(asteroid.x - sim.bodies[0].x, asteroid.y - sim.bodies[0].y) #distance relative to mc (sun)
                if position:
                    size = max((asteroid.size * WIDTH) / (viewport.half_size * 2), 1.5) 
                   # print(size)#change this - minimum size 3 for asteroids
                    pygame.draw.circle(bodies_surface, asteroid.color, position, size)
                

        
        
        screen.blit(orbits_surface, (0, 0))
        if asteroid_orbit_trail.visible:  # Only blit asteroid orbits if visible
            screen.blit(asteroid_orbits_surface, (0, 0))  #dahes 
        screen.blit(bodies_surface, (0, 0))

        
        gui_controls.draw_ui(screen)

        # Count asteroids for display
        num_asteroids = max(0, len(sim.asteroids))  # Total bodies minus Sun and 8 planets
        nea_count = getattr(sim, 'nea_count', 0)
        distant_count = getattr(sim, 'distant_count', 0)
    

        zoom_text = font.render(f"Zoom: {viewport.zoom_factor:.2f}x \nCenter: {viewport.center_x:.2f}, {viewport.center_y:.2f} \nHalf-size: {viewport.half_size:.2f}\nNEA: {nea_count}, Distant: {distant_count} ", True, (255, 255, 255))
        years_text = font1.render(f"no of years passed: {get_time(sim.t):.0f}", True, (255, 255, 255))
       #q controls_text = font_small.render("Controls: N=Show asteroid names, L=Load more asteroids", True, (200, 200, 200))\nAsteroids: {num_asteroids}
        
        screen.blit(zoom_text, (1150, 10))
        screen.blit(years_text, (10, 800))
        #screen.blit(controls_text, (10, 10))

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
            # gui starts here 
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
                elif event.key == pygame.K_o:
                    asteroid_orbit_trail.toggle_visibility()
                    # Clear the asteroid orbits surface when toggling off
                    if not asteroid_orbit_trail.visible:
                        asteroid_orbits_surface.fill((0, 0, 0, 0))
                    else:
                        # Redraw all asteroid trails when toggling back on
                        asteroid_orbits_surface.fill((0, 0, 0, 0))
                        for k, asteroid in enumerate(sim.asteroids):
                            if should_draw_asteroid_orbit(k, asteroid, sim):
                                trail_positions = asteroid_orbit_trail.get_trail(k)
                                lighter_color = lighten_color(asteroid.color, 0.2)  # Make asteroid orbits lighter
                                for trail_x, trail_y in trail_positions:
                                    trail_pixel_pos = viewport_to_pixels(trail_x, trail_y)
                                    if trail_pixel_pos:
                                        pygame.draw.circle(asteroid_orbits_surface, asteroid.color, trail_pixel_pos, 1)
                # New asteroid visibility controls
                elif event.key == pygame.K_1:  # Toggle NEA visibility
                    asteroid_visibility.nea_visible = not asteroid_visibility.nea_visible
                   # print(f"NEA asteroids {'visible' if asteroid_visibility.nea_visible else 'hidden'}")
                elif event.key == pygame.K_2:  # Toggle NEA orbits
                    asteroid_visibility.nea_orbits_visible = not asteroid_visibility.nea_orbits_visible
                   # print(f"NEA orbits {'visible' if asteroid_visibility.nea_orbits_visible else 'hidden'}")
                    # Trigger redraw of asteroid orbits surface
                    viewport.zoom_changed = True
                elif event.key == pygame.K_3:  # Toggle distant asteroid visibility
                    asteroid_visibility.distant_visible = not asteroid_visibility.distant_visible
                    #print(f"Distant asteroids {'visible' if asteroid_visibility.distant_visible else 'hidden'}")
                elif event.key == pygame.K_4:  # Toggle distant asteroid orbits
                    asteroid_visibility.distant_orbits_visible = not asteroid_visibility.distant_orbits_visible
                   # print(f"Distant orbits {'visible' if asteroid_visibility.distant_orbits_visible else 'hidden'}")
                    # Trigger redraw of asteroid orbits surface
                    viewport.zoom_changed = True
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
                    #dragging = True
                    mouse_pos = pygame.mouse.get_pos()

                    if gui_controls.info_bubble_visible:
       
                      bubble_rect = gui_controls.info_bubble.get_relative_rect()
                      bubble_rect.topleft = gui_controls.info_bubble.get_abs_rect().topleft
            
                      if not bubble_rect.collidepoint(mouse_pos):
               
                       gui_controls._hide_info_bubble()
     
                    else:
                      world_pos = pixels_to_viewport(mouse_pos[0], mouse_pos[1])
                      if not gui_controls.handle_object_click(mouse_pos, world_pos):
                         dragging = True
                         drag_start_pos = mouse_pos


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

def lighten_color(color, factor=0.6):
    """Lightening a color by blending it with white
    factor: 0.0 = original color, 1.0 = white
    """
    color = name_to_rgb(color)
    return (
        int(color[0] + (255 - color[0]) * factor),
        int(color[1] + (255 - color[1]) * factor),
        int(color[2] + (255 - color[2]) * factor)
    )


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