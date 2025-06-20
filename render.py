import time
from reboundsim import Simulation
import pygame
import cv2 as cv
import click
import math
import numpy as np




WIDTH = 1000
HEIGHT = 1000
VIEWPORT_HALF_SIZE = 3 #change this for closer look at inner planets (=3) (have to add zoom)
# VIEWPORT_HALF_SIZE = 31
# VIEWPORT_HALF_SIZE = 14
VIEWPORT = ((-VIEWPORT_HALF_SIZE, -VIEWPORT_HALF_SIZE), (VIEWPORT_HALF_SIZE, VIEWPORT_HALF_SIZE))

SIZE = WIDTH, HEIGHT
VIEWPORT_WIDTH = VIEWPORT[1][0] - VIEWPORT[0][0]
VIEWPORT_HEIGHT = VIEWPORT[1][1] - VIEWPORT[0][1]


def viewport_to_pixels(x, y):
    if x < VIEWPORT[0][0] or x > VIEWPORT[1][0]: 
        return None
    if y < VIEWPORT[0][1] or y > VIEWPORT[1][1]:
        return None

    px = ((x - VIEWPORT[0][0]) * WIDTH) / VIEWPORT_WIDTH
    py = ((VIEWPORT_HEIGHT - (y - VIEWPORT[0][1])) * HEIGHT) / VIEWPORT_HEIGHT

    return px, py

@click.command()
@click.option('--output_video', '-v', default='', help='Filename of video to write to')
def main(output_video):
    sim = Simulation()

    pygame.init()

    screen = pygame.display.set_mode(SIZE)
    orbits_surface = pygame.Surface(SIZE)
    bodies_surface = pygame.Surface(SIZE)
    bodies_surface.set_colorkey((0, 0, 0))
    #font = pygame.font.Font('freesansbold.ttf', 32)
    

    screen.fill((0, 0, 0))
  
    if output_video:
        pygame.display.update()  # force display to assume actual size
        video_writer = cv.VideoWriter(output_video, cv.VideoWriter_fourcc(*'DIVX'), 30, screen.get_size())

    delay = 0.025 # 25ms bw frames
    log_t = 0 # time tracker for logging
    month = 11
    week = 44


    paused = False 

    while True:
        t_start = time.time()

        bodies_surface.fill((0, 0, 0))

    #time to render the celestial balls :p
        for body in sim.bodies:
         position = viewport_to_pixels(body.x - sim.bodies[0].x, body.y - sim.bodies[0].y) #distance relative to mc (sun)
         if position:
                size = max((body.size * WIDTH) / (VIEWPORT_HALF_SIZE*2), 5) #change this
                pygame.draw.circle(orbits_surface, body.color, position, 1)
                pygame.draw.circle(bodies_surface, body.color, position, size)
            
        screen.blit(orbits_surface, (0, 0))  #dahes 
        screen.blit(bodies_surface, (0, 0))

        sun_distance = calculate_dist(sim.bodies[0], sim.bodies[3])

        #logging
        if get_time(sim.t) > log_t: 
            print(f't={get_time(sim.t):.2f}, month {int((week-1)/4.3)+1:2d}, week {week:2d}: {sun_distance}')
            log_t += 1/52
            if week == 52:
                week = 1
            else:
                week += 1

        pygame.display.update()

        if output_video:
            video_writer.write(image_from_screen(screen))

        if not paused:
            sim.iterate() #sim step one at a time

            t_delta = time.time() - t_start    #frame rate contrpl
            if t_delta < delay:
                time.sleep(delay - t_delta)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if output_video:
                    video_writer.release()
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_s:
                    save_screenshot(screen, sim.t)
                elif event.key == pygame.K_q:
                    if output_video:
                        video_writer.release()
                    pygame.quit()
                    quit()


def save_screenshot(screen, t):
    image = image_from_screen(screen)
    t_str = format_time(t).replace('.', '_')
    filename = f'screenshot_t_{t_str}.png'
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








