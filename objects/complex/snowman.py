import numpy as np
from utils import transforms
from objects import Sphere, Cone
from utils.material import Material
from objects.complex import ObjectComplex


SNOW_BALL = Material()
CARROT = Material(color=np.array([226., 146., 100.]), shininess=16)
BUTTON = Material(color=np.array([30., 30., 30.]), shininess=0.005)


class Snowman(ObjectComplex):
    def __init__(self, position: np.ndarray):
        self.axis = np.array([0., 1., 0.])
        self.heights = [0.8, 2.0, 3.05, 3.1, 3.2, 3.2]

        head = Sphere(np.array([0., 3.05, 0.]), 0.5, material=SNOW_BALL)
        body_up = Sphere(np.array([0., 2.0, 0.]), 0.8, material=SNOW_BALL)

        buttons = []
        buttons_ray = np.array([0., 0., -1.])*body_up.radius
        angle = np.pi/10
        buttons_ray = transforms.rotateX(buttons_ray, angle*2)
        for _ in range(5):
            buttons.append(Sphere(body_up.position + buttons_ray, 0.08, material=BUTTON))
            self.heights.append(buttons_ray[1] + body_up.position[1])
            buttons_ray = transforms.rotateX(buttons_ray, -angle)


        super().__init__(
            position,
            [
                Sphere(np.array([0., 0.8, 0.]), 0.8, material=SNOW_BALL),
                body_up,
                head,
                Cone(np.array([0., 3.1, -1.]), np.array([0., 0., -1.]), 0.6, .1, material=CARROT),
                Sphere(np.array([0.2, 3.2, -0.4]), 0.08, material=BUTTON),
                Sphere(np.array([-0.2, 3.2, -0.4]), 0.08, material=BUTTON),
                *buttons
            ]
        )
        self.isComplex = True
        for part in self.parts:
            part.superObject = self
            part.translate(position)

    def rotate(self, angle: float, axis: np.ndarray, around: np.ndarray = None):
        if around is not None:
            self.position = around + transforms.rotate(self.position - around, angle, axis)
            for part in self.parts:
                part.rotate(angle, axis, around)
        else:
            self.axis = transforms.rotate(self.axis, angle, axis)
            for index, part in enumerate(self.parts):
                part.rotate(angle, axis, None)
                part.rotate(angle, axis, self.position + self.axis * self.heights[index])
