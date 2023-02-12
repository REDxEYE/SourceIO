REPARENT_LOGS_NONE = 0  # Do not touch logs
REPARENT_LOGS_OVERWRITE = 1  # Overwrite the logs with new local transform of the dag
REPARENT_LOGS_OFFSET_LOCAL = 2  # Apply the transform required to maintain the world space transform of the dag to all log samples
REPARENT_LOGS_MAINTAIN_WORLD = 3  # Modify the logs so that the world space position and orientation animation is maintained

from mathutils import Color as BColor
from mathutils import Vector as BVector

from . import mathlib


class Vector:

    def __init__(self, x, y, z):
        self.v = BVector([x, y, z])

    def NormalizeInPlace(self):
        self.v.normalize()

    @property
    def x(self):
        return self.v.x

    @x.setter
    def x(self, value):
        self.v.x = value

    @property
    def y(self):
        return self.v.y

    @y.setter
    def y(self, value):
        self.v.y = value

    @property
    def z(self):
        return self.v.z

    @z.setter
    def z(self, value):
        self.v.z = value

    def __neg__(self):
        return Vector(*(-self.v))


class Color:
    def __init__(self, r, g, b, a):
        self.c = BColor([r/255, g/255, b/255])
