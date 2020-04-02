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


class Vtex:

    def __init__(self, vtex_path):
        self.valve_file = ValveFile(vtex_path)
        self.valve_file.read_block_info()

    def load(self):
        name = Path(self.valve_file.filename).stem
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        rgb, alpha = data_block.get_rgb_and_alpha()

        image = bpy.data.images.new(
            name + '_RGB',
            width=data_block.width,
            height=data_block.height)
        image.pixels = np.divide(list(rgb), 255)
        image.pack()
        image = bpy.data.images.new(
            name + '_A',
            width=data_block.width,
            height=data_block.height)
        image.pixels = np.divide(list(alpha), 255)
        image.pack()
