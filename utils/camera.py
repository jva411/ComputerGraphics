import numba
import ctypes
import numpy as np
from utils.ray import Ray
from objects import Object
from utils import transforms
import multiprocessing as mp

BLACK = np.array([0., 0., 0.])
SKY_COLOR = np.array([203., 224., 233.])

class Camera():
    def __init__(self, resolution: tuple[int, int], position: np.ndarray, at: np.ndarray, rotation=0, distance=1., perpendicular=False, n_threads=1, windowSize=None, debounces=0, n_samples=1, gamma_correction=False):
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
        self.pixel_width, self.pixel_height = (self.windowSize / 100.) / self.resolution
        self.distance = distance
        self.frameOrigin = self.position + self.direction * self.distance
        self.n_threads = n_threads
        self.debounces = debounces
        self.n_samples = n_samples
        self.gamma_correction = gamma_correction

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

        self.pixelPositions = calcuate_pixels(self.frameOrigin, self.up, self.right, *self.resolution, self.pixel_width, self.pixel_height)
        # self.rayDirections = calcuate_directions(self.position, self.pixelPositions, *self.resolution)

    def shared_array_to_numpy_array(self, shared_array):
        return np.ctypeslib.as_array(shared_array).reshape((*self.resolution[::-1], 3))

    def init_worker(self, shared_buffer, shared_progress):
        global buffer
        global progress
        buffer = self.shared_array_to_numpy_array(shared_buffer)
        progress = np.ctypeslib.as_array(shared_progress).reshape(1)

    def getRay(self, x, y):
        random_x, random_y = (np.random.random(2) - 0.5) * np.array([self.pixel_width, self.pixel_height])
        random_pixel_pos = self.pixelPositions[x, y] + self.right * random_x + self.up * random_y
        if self.perpendicular:
            return Ray(random_pixel_pos, self.direction)

        random_direction = transforms.normalize(random_pixel_pos - self.position)
        return Ray(self.position, random_direction)

    def threadedRayCast(self, x0, y, batch):
        samples_buffer = np.zeros((self.n_samples, 3), dtype=np.float64)
        for obj in self.scene.objects:
            obj.preCalc()

        width, _ = self.resolution
        for x in range(x0, min(x0 + batch, width)):
            for sample in range(self.n_samples):
                ray = self.getRay(x, y)
                target, lightness = self.calcRecursiveRayCast(ray, self.debounces, 1)

                if target is None:
                    samples_buffer[sample] = SKY_COLOR
                else:
                    sample_color = np.clip(lightness, 0., 255.)
                    if self.gamma_correction:
                        sample_color = gamma_correction(sample_color)

                    samples_buffer[sample] = sample_color

            buffer[y, x] = np.mean(samples_buffer, axis=0)
            progress[0] += 1

    def calcRecursiveRayCast(self, ray: Ray, debounces=0, depth=1) -> tuple[Object, np.ndarray]:
        weight = 0.5 ** depth
        point, target = self.scene.rayTrace(ray)
        if target is None or (weight < 0.5 and np.random.random() < 0.5):
            return None, None

        normal = target.getNormal(point)
        lightness = self.scene.computeLightness(point, normal, ray, target)
        if debounces > 0:
            scattered_ray, weight = target.material.scatter(ray, point, normal)
            if scattered_ray is not None:
                _, scattered_color = self.calcRecursiveRayCast(scattered_ray, debounces - 1, depth+1)
                if scattered_color is not None:
                    lightness += scattered_color * weight

        return target, lightness

    def rayCast(self, scene=None):
        arraySize = self.resolution[0] * self.resolution[1] * 3

        shared_buffer = mp.Array(ctypes.c_float, int(arraySize), lock=False)
        shared_progress = mp.Array(ctypes.c_int, 1, lock=False)
        self.buffer = self.shared_array_to_numpy_array(shared_buffer)
        self.shared_progress = np.ctypeslib.as_array(shared_progress).reshape(1)
        self.shared_progress[0] = 0
        if scene is not None:
            scene.image = self.buffer
        self.scene = scene

        width, height = self.resolution
        batch = 64
        pixels = ((x0, height-y-1) for y in range(height) for x0 in range(0, width, batch))
        args = (pixel + (batch,) for pixel in pixels)

        pool = mp.Pool(processes=self.n_threads, initializer=self.init_worker, initargs=(shared_buffer, shared_progress))

        pool.starmap(self.threadedRayCast, args)


@numba.jit
def gamma_correction(color: np.ndarray):
    normalized_color = color / 255.

    corrected_color = np.sqrt(normalized_color)

    return corrected_color * 255.


@numba.jit
def calcuate_pixels(frameOrigin, viewUp, viewRight, width, height, pixel_width, pixel_height):
    points = np.zeros((width, height, 3))

    for x, y in np.ndindex((width, height)):
        dx = (x - width/2) * pixel_width + pixel_width/2
        dy = (y - height/2) * pixel_height + pixel_height/2
        points[x, y] = frameOrigin + dy*viewUp + dx*viewRight

    return points


# @numba.jit
# def calcuate_directions(eye, pixels, width, height):
#     directions = np.zeros((width, height, 3))
#     for x, y in np.ndindex((width, height)):
#         direction = pixels[x, y] - eye
#         directions[x, y] = direction/np.linalg.norm(direction)

#     return directions
