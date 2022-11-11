import cv2
import pygame
import numpy as np
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
        self.buttons = {
            pygame.BUTTON_LEFT: self.pick
        }
        self.selected = None
        self.updateSelected = False
        self.closed = False

    def open(self):
        pygame.display.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        self.font = pygame.font.SysFont(None, 16)

        pygame.display.set_caption(self.title)

    def startLoop(self):
        while True:
            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        self.close()
                    case pygame.KEYDOWN:
                        if event.key in self.keys:
                            self.keys[event.key]()
                    case pygame.MOUSEBUTTONUP:
                        if event.button in self.buttons:
                            self.buttons[event.button]()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.scene.update()
            self.update()
            pygame.display.flip()

            if (self.closed):
                pygame.quit()
                exit(0)
            pygame.time.wait(10)

    def update(self):
        if self.updateSelected:
            self.renderSelectedProps()

    def close(self):
        self.closed = True

    def screenshot(self):
        cv2.imwrite(
            f'screenshots/{dt.now().strftime("%Y-%m-%d_%H-%M-%S")}.png',
            cv2.flip(self.scene.camera.buffer[...,::-1], 0),
        )

    def renderSelectedProps(self):
        if self.updateSelected is False: return

        self.updateSelected = False
        self.scene.image = self.scene.camera.buffer.copy()
        if self.selected is None: return

        for idx, line in enumerate(self.selected.getDescription().split('\n')):
            cv2.putText(self.scene.image, line, (5, self.scene.height - 15*(idx+1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 1, cv2.LINE_AA, True)

    def pick(self):
        [x, y] = pygame.mouse.get_pos()
        occurrences = {}
        width, height = self.scene.camera.resolution
        for dx, dy in np.ndindex((7, 7)):
            dx -= 3
            dy -= 3
            if (
                0 <= x + dx < width and
                0 <= y + dy < height
            ):
                _, obj = self.scene.rayTrace(self.scene.camera.getRay(x+dx, height - (y+dy)))

                if obj is not None: occurrences[obj] = occurrences.get(obj, 0) + 1

        self.updateSelected = True
        obj = max(occurrences, key=lambda x: occurrences[x], default=None)
        if obj is None:
            self.selected = None
            return

        while obj.superObject is not None and (not obj.superObject.isBVH or obj.superObject.superObject is not None):
            obj = obj.superObject

        self.selected = obj
