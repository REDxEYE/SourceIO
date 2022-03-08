from typing import List

from .redi_block_types import *
from .base_block import DataBlock
from ..utils.binary_keyvalue import BinaryKeyValue

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


class RED2(DataBlock):

    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)
        self.blocks = []  # type:List[Dependencies]

    def read(self):
        reader = self.reader
        vkv3 = BinaryKeyValue(self.info_block)
        vkv3.read(reader)
        block = AdditionalInputDependencies()
        for aid in vkv3.kv.get('m_AdditionalInputDependencies', []):
            dependency = block.dependency()
            dependency.content_search_path = aid['m_SearchPath']
            dependency.content_relative_name = aid['m_RelativeFilename']
            dependency.file_crc = aid['m_nFileCRC']
            block.container.append(dependency)
        block.size = len(block.container)
        self.blocks.append(block)

        block = SpecialDependencies()
        for aid in vkv3.kv.get('m_SpecialDependencies', []):
            dependency = block.dependency()
            dependency.compiler_identifier = aid['m_CompilerIdentifier']
            dependency.string = aid['m_String']
            dependency.user_data = aid['m_nUserData']
            dependency.fingerprint = aid['m_nFingerprint']
            block.container.append(dependency)
        block.size = len(block.container)
        self.blocks.append(block)

        block = ArgumentDependencies()
        for aid in vkv3.kv.get('m_ArgumentDependencies', []):
            dependency = block.dependency()
            dependency.parameter_name = aid['m_ParameterName']
            dependency.parameter_type = aid['m_ParameterType']
            dependency.fingerprint = aid['m_nFingerprint']
            dependency.fingerprint_default = aid['m_nFingerprintDefault']
            block.container.append(dependency)
        block.size = len(block.container)
        self.blocks.append(block)
