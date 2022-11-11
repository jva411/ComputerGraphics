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
    def __init__(self, resolution: tuple[int, int], position: np.ndarray, at: np.ndarray, ratio = np.array([4., 3.]), rotation=0, perpendicular=False):
        from utils.scene import Scene
        self.scene: Scene = None

        self.position = position
        self.direction = transforms.normalize(at - position)
        self.resolution = np.array([*resolution], dtype=np.int32)
        self.ratio = ratio
        self.buffer = np.zeros((*resolution[::-1], 3), dtype=np.float64)
        self.perpendicular = perpendicular
        self.rx, self.ry = self.ratio / self.resolution
        self.frameOrigin = self.position.copy()
        if not self.perpendicular: self.frameOrigin += self.direction * 5

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
        if rotation > 0:
            self.up = transforms.rotate(self.up, np.radians(rotation), self.direction)
            self.right = transforms.rotate(self.right, np.radians(rotation), self.direction)

        self.pixelPositions = calcuate_pixels(self.frameOrigin, self.up, self.right, *self.resolution, self.rx, self.ry)
        self.rayDirections = calcuate_directions(self.position, self.pixelPositions, *self.resolution)

    def to_numpy_array(self, shared_array, shape):
        '''Create a numpy array backed by a shared memory Array.'''
        array = np.ctypeslib.as_array(shared_array)
        return array.reshape(shape)

    def init_worker(self, shared_buffer, shape):
        '''
        Initialize worker for processing:
        Create the numpy array from the shared memory Array for each process in the pool.
        '''
        global buffer
        buffer = self.to_numpy_array(shared_buffer, shape)

    def getRay(self, x, y):
        if self.perpendicular:
            return Ray(self.pixelPositions[x, y], self.direction)

        return Ray(self.position, self.rayDirections[x, y])

    def rayCast2(self, x0, y0, shape):
        for x, y in np.ndindex(*shape):
            x += x0
            y += y0
            ray = self.getRay(x, y)
            point, target = self.scene.rayTrace(ray)

            if target is None:
                buffer[y, x] = [203, 224, 233]
            else:
                normal = target.getNormal(point,)
                lightness = self.scene.computeLightness(point, normal, ray, target)
                buffer[y, x] = np.clip(target.getColor(point) * lightness, 0., 255.)

    def rayCast(self, scene=None):
        arraySize = self.resolution[0] * self.resolution[1] * 3
        n = 2
        [rw, rh] = np.array(self.resolution // n, dtype=np.uint32)
        x0 = self.resolution[0] % n
        y0 = self.resolution[1] % n

        shared_buffer = mp.Array(ctypes.c_float, int(arraySize), lock=False)
        self.buffer = self.to_numpy_array(shared_buffer, (*self.resolution[::-1], 3))
        if scene is not None: scene.image = self.buffer
        pool = mp.Pool(processes=n*n, initializer=self.init_worker, initargs=(shared_buffer, (*self.resolution[::-1], 3)))

        buffers = []
        for i, j in np.ndindex(n, n):
            w = (1 if x0>0 else 0) + rw
            h = (1 if y0>0 else 0) + rh
            x0 -= 1
            y0 -= 1
            buffers.append((i * w, j * h, (w, h)))

        pool.starmap(self.rayCast2, buffers)


@numba.jit
def calcuate_pixels(frameOrigin, viewUp, viewRight, width, height, rx, ry):
    points = np.zeros((width, height, 3))
    for x, y in np.ndindex((width, height)):
        dx = (x - width/2) * rx
        dy = (y - height/2) * ry
        points[x, y] = frameOrigin + dy*viewUp + dx*viewRight

    return points

@numba.jit
def calcuate_directions(eye, pixels, width, height):
    directions = np.zeros((width, height, 3))
    for x, y in np.ndindex((width, height)):
        direction = pixels[x, y] - eye
        directions[x, y] = direction/np.linalg.norm(direction)

    return directions
