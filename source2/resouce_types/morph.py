from ..blocks import MRPH
from . import ValveCompiledResource


class ValveCompiledMorph(ValveCompiledResource):
    data_block_class = MRPH

    def __init__(self, path_or_file):
        super().__init__(path_or_file)
