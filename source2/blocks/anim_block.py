from .data_block import DATA
import numpy as np


class ANIM(DATA):

    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)

    def read(self):
        super().read()


