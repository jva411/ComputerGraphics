import math
import numba
import ctypes
import numpy as np
from utils.ray import Ray
from objects import Object
from utils import transforms
import multiprocessing as mp
from lights.lights import Light
from multiprocessing import Process, Pool

class Camera():
    def __init__(self, resolution: tuple[int, int], position: np.ndarray, at: np.ndarray, rotation=0, distance=5., perpendicular=False, n_threads=1, windowSize=None):
        from utils.scene import Scene
        self.scene: Scene = None

        self.position = position
        self.at = at
        self.direction = transforms.normalize(at - position)
        self.resolution = np.array([*resolution], dtype=np.int32)
        self.buffer = np.zeros((*resolution[::-1], 3), dtype=np.float64)
        self.perpendicular = perpendicular
        self.windowSize = self.resolution
        if windowSize is not None: self.windowSize = np.array([*windowSize], dtype=np.int32)
        self.rx, self.ry = (self.windowSize / 100.) / self.resolution
        self.distance = distance
        self.frameOrigin = self.position + self.direction * self.distance
        self.n_threads = n_threads

        dirXZ = self.direction[[0, 2]]
        if all(dirXZ == np.array([0., 0.])):
            aXZ = 0
        else:
            rDirXZ = transforms.rotate2D(transforms.normalize(dirXZ), -np.pi/2)
            dX = rDirXZ @ np.array([1., 0.])
            dZ = rDirXZ @ np.array([0., 1.])
            aXZ = np.arccos(dX)
            if (dZ < 0): aXZ = 2*np.pi - aXZ

        self.right = transforms.rotateY(np.array([1., 0., 0.]), -aXZ)
        self.up = transforms.rotate(self.direction, -np.pi/2, self.right)
        self.rotation = rotation
        if rotation > 0:
            self.up = transforms.rotate(self.up, np.radians(rotation), self.direction)
            self.right = transforms.rotate(self.right, np.radians(rotation), self.direction)

        self.pixelPositions = calcuate_pixels(self.frameOrigin, self.up, self.right, *self.resolution, self.rx, self.ry)
        self.rayDirections = calcuate_directions(self.position, self.pixelPositions, *self.resolution)

    def shared_array_to_numpy_array(self, shared_array):
        return np.ctypeslib.as_array(shared_array).reshape((*self.resolution[::-1], 3))

    def init_worker(self, shared_buffer):
        global buffer
        buffer = self.shared_array_to_numpy_array(shared_buffer)

    def getRay(self, x, y, sample):
        if self.perpendicular:
            return Ray(self.pixelPositions[x, y, sample], self.direction)

        return Ray(self.position, self.rayDirections[x, y, sample])

    def threadedRayCast(self, x0, y, batch):
        samples_buffer = np.zeros((4, 3), dtype=np.float64)
        for obj in self.scene.objects:
            obj.preCalc()

        width, _ = self.resolution
        for x in range(x0, min(x0 + batch, width)):
            for sample in range(4):
                ray = self.getRay(x, y, sample)
                point, target = self.scene.rayTrace(ray)

                if target is None:
                    samples_buffer[sample] = [203, 224, 233]
                else:
                    normal = target.getNormal(point)
                    lightness = self.scene.computeLightness(point, normal, ray, target)
                    samples_buffer[sample] = np.clip(target.getColor(point) * lightness, 0., 255.)

            buffer[y, x] = np.mean(samples_buffer, axis=0)

    def rayCast(self, scene=None):
        arraySize = self.resolution[0] * self.resolution[1] * 3

        shared_buffer = mp.Array(ctypes.c_float, int(arraySize), lock=False)
        self.buffer = self.shared_array_to_numpy_array(shared_buffer)
        if scene is not None:
            scene.image = self.buffer
        self.scene = scene

        width, height = self.resolution
        batch = 64
        pixels = ((x0, height-y-1) for y in range(height) for x0 in range(0, width, batch))
        args = (pixel + (batch,) for pixel in pixels)

        pool = mp.Pool(processes=self.n_threads, initializer=self.init_worker, initargs=(shared_buffer,))

        print('resolution:', self.resolution)
        print('windowSize:', self.windowSize)
        print('rx, ry:', self.rx, self.ry)
        print('buffer shape:', self.buffer.shape)
        pool.starmap(self.threadedRayCast, args)


@numba.jit
def calcuate_pixels(frameOrigin, viewUp, viewRight, width, height, rx, ry):
    points = np.zeros((width, height, 4, 3))
    samples = ((rx/2, ry/4), (rx/2, ry*3/4), (rx/4, ry/2), (rx*3/4, ry/2))
    for x, y, sample in np.ndindex((width, height, np.int32(4))):
            sample_x, sample_y = samples[sample]
            dx = (x - width/2) * rx + sample_x
            dy = (y - height/2) * ry + sample_y
            points[x, y, sample] = frameOrigin + dy*viewUp + dx*viewRight

    return points

@numba.jit
def calcuate_directions(eye, pixels, width, height):
    directions = np.zeros((width, height, 4, 3))
    for x, y, sample in np.ndindex((width, height, np.int32(4))):
        direction = pixels[x, y, sample] - eye
        directions[x, y, sample] = direction/np.linalg.norm(direction)

    return directions
