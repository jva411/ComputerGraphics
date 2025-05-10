import pygame
import numpy as np


class Window:
    def __init__(self, width, height, render):
        self.width = width
        self.height = height
        self.render = render
        self.tick = 0

    def open(self):
        pygame.display.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 16)

        pygame.display.set_caption("Ray Tracer using CUDA")

    def startLoop(self):
        try:
            self.loop()
        except KeyboardInterrupt:
            self._close()

    def loop(self):
        while True:
            image = self.render(self.tick)

            self.screen.blit(pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)) * 255.), (0, 0))
            self.screen.blit(self.font.render(str(int(self.clock.get_fps())), True, (255, 255, 255)), (0, 0))
            pygame.display.flip()
            self.tick += 1
            self.clock.tick(60)


    def _close(self):
        pygame.quit()
