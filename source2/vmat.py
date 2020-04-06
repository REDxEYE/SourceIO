import os.path
import random
from pathlib import Path
from typing import List

import bpy
import math
import numpy as np
from mathutils import Vector, Matrix, Quaternion, Euler

from SourceIO.source2.blocks.common import SourceVector, SourceVertex
from .source2 import ValveFile


class Vmat:

    def __init__(self, vtex_path):
        self.valve_file = ValveFile(vtex_path)
        self.valve_file.read_block_info()

    def load(self):
        name = Path(self.valve_file.filename).stem
