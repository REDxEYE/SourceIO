from .redi_block_types import *
from .dummy import DataBlock

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

    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)
        self.blocks = []  # type:List[Dependencies]

    def read(self):
        reader = self.reader
        for redi_block in redi_blocks:
            block = redi_block()
            block.read(reader)
            self.blocks.append(block)
