import numpy as np
from vector import v3, normalize, rotate, rotate2D, rotateY

type Size = tuple[int, int]
type Camera = tuple[v3, v3, Size, v3, v3]

position: np.ndarray
direction: np.ndarray
resolution: np.ndarray
viewport_size: np.ndarray
pixel_size: np.ndarray
pixel00_position: np.ndarray
distance: float
right: np.ndarray
up: np.ndarray

def init(new_resolution: tuple[int, int], new_position: np.ndarray, at: np.ndarray, rotation=0, new_distance=5., perpendicular=False, n_threads=1, new_viewport_size=None, debounces=0, super_samples=False):
    global position, direction, resolution, viewport_size, pixel_size, pixel00_position, distance, right, up
    position = new_position
    direction = normalize(at - position)
    resolution = np.array([*new_resolution], dtype=np.int32)
    # self.buffer = np.zeros((*resolution[::-1], 3), dtype=np.float64)
    # self.perpendicular = perpendicular
    viewport_size = resolution if new_viewport_size is None else np.array([*new_viewport_size], dtype=np.int32)

    pixel_size = (viewport_size / 100.) / resolution
    distance = new_distance
    # self.frameOrigin = self.position + self.direction * self.distance
    # self.debounces = debounces
    # self.super_samples = super_samples

    directionXZ = direction[[0, 2]]
    if all(directionXZ == np.array([0., 0.])):
        aXZ = 0
    else:
        rotatedDirectionXZ = rotate2D(normalize(directionXZ), -np.pi/2)
        dX = rotatedDirectionXZ @ np.array([1., 0.])
        dZ = rotatedDirectionXZ @ np.array([0., 1.])
        aXZ = np.arccos(dX)
        if (dZ < 0):
            aXZ = 2*np.pi - aXZ

    right = rotateY(np.array([1., 0., 0.]), -aXZ)
    if rotation > 0:
        right = rotate(right, np.radians(rotation), direction)

    up = rotate(direction, -np.pi/2, right)

    pixel00_position = position + (direction * distance)
    pixel00_position = pixel00_position - (right * (resolution[0] * pixel_size[0] - pixel_size[0])/2)
    pixel00_position = pixel00_position + (up * (resolution[1] * pixel_size[1] - pixel_size[1])/2)

    # self.pixelPositions = calcuate_pixels(self.frameOrigin, self.up, self.right, *self.resolution, self.rx, self.ry, self.super_samples)
    # self.rayDirections = calcuate_directions(self.position, self.pixelPositions, *self.resolution, self.super_samples)

def get_device_camera():
    return (
        position,
        pixel00_position,
        (*pixel_size, 0.),  # Homogeneous properties
        up,
        right,
    )