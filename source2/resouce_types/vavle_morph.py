from pathlib import Path

# noinspection PyUnresolvedReferences
import bpy
import numpy as np

from ..blocks import MRPH
from ..source2 import ValveCompiledFile


class ValveCompiledMorph(ValveCompiledFile):
    data_block_class = MRPH

    def __init__(self, path_or_file):
        super().__init__(path_or_file)
