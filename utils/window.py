import cv2
import pygame
from utils.scene import Scene
from datetime import datetime as dt
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT


class Window:
    def __init__(self, scene: Scene, title="Title"):
        self.scene = scene
        self.title = title
        self.display = (scene.width, scene.height)
        self.keys = {
            pygame.K_ESCAPE: self.close,
            pygame.K_p: self.screenshot
        }

    def open(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption(self.title)

    def startLoop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                elif event.type == pygame.KEYDOWN:
                    if event.key in self.keys:
                        self.keys[event.key]()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.scene.update()
            pygame.display.flip()

            pygame.time.wait(10)  # TODO(udpate to ticks)

    def close(self):
        pygame.quit()
        quit()

    def screenshot(self):
        cv2.imwrite(
            f'screenshots/{dt.now().strftime("%d-%m-%Y_%H-%M-%S")}.png',
            cv2.flip(self.scene.camera.buffer[...,::-1], 0),
        )
