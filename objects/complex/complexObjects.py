import numpy as np
from objects.object import Object


class ComplexObject(Object):
    def __init__(self, position: np.ndarray, parts: list[Object]):
        super().__init__(position, None, 1.)
        self.parts = parts
