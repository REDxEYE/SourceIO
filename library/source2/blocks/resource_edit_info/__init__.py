from dataclasses import dataclass

from SourceIO.library.utils import Buffer

from SourceIO.library.source2.blocks.base import BaseBlock
from SourceIO.library.source2.keyvalues3.binary_keyvalues import read_valve_keyvalue3
from .dependencies import *
from SourceIO.library.source2.utils.ntro_reader import NTROBuffer


@dataclass
class ResourceEditInfo(BaseBlock):
    inputs: InputDependencies
    additional_inputs: AdditionalInputDependencies
    arguments: ArgumentDependencies
    special_deps: SpecialDependencies
    custom_deps: CustomDependencies
    additional_files: AdditionalRelatedFiles
    child_resources: ChildResources
    extra_ints: ExtraInts
    extra_floats: ExtraFloats
    extra_strings: ExtraStrings

    def __str__(self) -> str:
        return f"<Resource edit info:" \
               f" inputs:{len(self.inputs)}," \
               f" arguments:{len(self.arguments)}," \
               f" child resources:{len(self.child_resources)}>"

    @classmethod
    def from_buffer(cls, buffer: NTROBuffer):
        return cls(
            InputDependencies.from_buffer(buffer),
            AdditionalInputDependencies.from_buffer(buffer),
            ArgumentDependencies.from_buffer(buffer),
            SpecialDependencies.from_buffer(buffer),
            CustomDependencies.from_buffer(buffer),
            AdditionalRelatedFiles.from_buffer(buffer),
            ChildResources.from_buffer(buffer),
            ExtraInts.from_buffer(buffer),
            ExtraFloats.from_buffer(buffer),
            ExtraStrings.from_buffer(buffer),
        )


class ResourceEditInfo2(ResourceEditInfo):
    @classmethod
    def from_buffer(cls, buffer: Buffer):
        vkv = read_valve_keyvalue3(buffer)
        return cls(
            InputDependencies.from_vkv3(vkv['m_InputDependencies']),
            AdditionalInputDependencies.from_vkv3(vkv['m_AdditionalInputDependencies']),
            ArgumentDependencies.from_vkv3(vkv['m_ArgumentDependencies']),
            SpecialDependencies.from_vkv3(vkv['m_SpecialDependencies']),
            CustomDependencies(),
            AdditionalRelatedFiles.from_vkv3(vkv['m_AdditionalRelatedFiles']),
            ChildResources.from_vkv3(vkv['m_ChildResourceList']),
            ExtraInts(),
            ExtraFloats(),
            ExtraStrings()

        )
