import numba
import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from objects.object import Object, t_correction


class dotdict(dict):
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)

    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

    def __hash__(self):
        return iterHash(self.keys()) + iterHash(self.values())

def iterHash(iterable):
    h = 0
    for x in iterable:
        if not isinstance(x, str) and hasattr(x, '__iter__'): h += iterHash(x)
        else: h += hash(x)
    return h

class Sphere(Object):
    def __init__(self, position: np.ndarray, radius: float, material = BLANK):
        super().__init__(position, material)
        self.radius = radius


    def intersects(self, ray: Ray) -> np.ndarray:
        return intersects(ray, self.position, self.radius)

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return (point - self.position) / self.radius

    def getPhysic(self):
        return dotdict({
            'isComplex': False,
            'isBVH': False,
            'intersects': intersects,
            'params': (self.position, self.radius)
        })


@numba.jit(cache=True)
def intersects(ray, position, radius):
    co = ray.origin - position

    b = 2 * co @ ray.direction
    c = co @ co - radius ** 2
    delta = b ** 2 - 4*c
    if delta < 0: return None

    ts = []
    t1 = (-b + np.sqrt(delta)) / 2 - t_correction
    t2 = (-b - np.sqrt(delta)) / 2 - t_correction
    if 0 < t1 < ray.t:
        ts.append(t1)
    if 0 < t2 < ray.t:
        ts.append(t2)
    if len(ts) == 0:
        return None

    t = ts[0]
    if len(ts) > 1 and ts[1] < t:
        t = ts[1]

    ray.t = t
    return ray.origin + ray.direction*t
