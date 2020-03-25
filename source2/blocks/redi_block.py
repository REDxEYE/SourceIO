from typing import List

from .header_block import InfoBlock
from .redi_block_types import *
from .dummy import DataBlock
from ..source2 import ValveFile

redi_blocks = [InputDependencies,
               AdditionalInputDependencies,
               ArgumentDependencies,
               SpecialDependencies,
               CustomDependencies,
               AdditionalRelatedFiles,
               ChildResourceList,
               ExtraIntData,
               ExtraFloatData,
               ExtraStringData
               ]


class REDI(DataBlock):

    def __init__(self, valve_file: ValveFile, info_block):
        super().__init__(valve_file, info_block)
        self.blocks = []  # type:List[Dependencies]

    def read(self):
        reader = self.reader
        for redi_block in redi_blocks:
            block = redi_block()
            entry = reader.tell()
            block.offset = reader.read_int32()
            block.size = reader.read_int32()
            with reader.save_current_pos():
                reader.seek(entry + block.offset)
                block.read(reader)
                self.blocks.append(block)
        self.empty = False
