import pygame
import numpy as np
from numba import njit
from typing import Callable

width: int
height: int
render: Callable
screen: pygame.Surface
clock: pygame.time.Clock
font: pygame.font.Font


def init(new_width, new_height, new_render):
    global width, height, render
    width = new_width
    height = new_height
    render = new_render

def open():
    global screen, clock, font
    pygame.display.init()
    pygame.font.init()

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 16)

    pygame.display.set_caption("Ray Tracer using CUDA")

def startLoop():
    try:
        _loop()
    except KeyboardInterrupt:
        _close()

def _loop():
    tick = 0
    while True:
        image = render(tick)

        screen.blit(pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)) * 255.), (0, 0))
        screen.blit(font.render(str(int(clock.get_fps())), True, (255, 255, 255)), (0, 0))
        pygame.display.flip()
        tick += 1
        clock.tick(60)

def _close():
    pygame.quit()
