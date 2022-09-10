import numpy as np
from utils.material import BLANK
from objects.object import Object


class ComplexObject(Object):
    def __init__(self, position: np.ndarray, parts: list[Object]):
        super().__init__(position, BLANK)
        self.parts = parts
