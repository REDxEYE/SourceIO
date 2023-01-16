from .....utils import Buffer
from ....resource_types.resource import CompiledResource
from ...blocks.base import BaseBlock
from ...keyvalues3.binary_keyvalues import BinaryKeyValues
from .dependencies import *


class ResourceEditInfo(BaseBlock):
    def __init__(self, buffer: Buffer, resource: CompiledResource):
        super().__init__(buffer, resource)
        self.inputs = InputDependencies()
        self.additional_inputs = AdditionalInputDependencies()
        self.arguments = ArgumentDependencies()
        self.special_deps = SpecialDependencies()
        self.custom_deps = CustomDependencies()
        self.additional_files = AdditionalRelatedFiles()
        self.child_resources = ChildResources()
        self.extra_ints = ExtraInts()
        self.extra_floats = ExtraFloats()
        self.extra_strings = ExtraStrings()

    def __str__(self) -> str:
        return f"<Resource edit info:" \
               f" inputs:{len(self.inputs)}," \
               f" arguments:{len(self.arguments)}," \
               f" child resources:{len(self.child_resources)}>"

    @classmethod
    def from_buffer(cls, buffer: Buffer, resource: CompiledResource):
        self = cls(buffer, resource)
        self.inputs = InputDependencies.from_buffer(buffer)
        self.additional_inputs = AdditionalInputDependencies.from_buffer(buffer)
        self.arguments = ArgumentDependencies.from_buffer(buffer)
        self.special_deps = SpecialDependencies.from_buffer(buffer)
        self.custom_deps = CustomDependencies.from_buffer(buffer)
        self.additional_files = AdditionalRelatedFiles.from_buffer(buffer)
        self.child_resources = ChildResources.from_buffer(buffer)
        self.extra_ints = ExtraInts.from_buffer(buffer)
        self.extra_floats = ExtraFloats.from_buffer(buffer)
        self.extra_strings = ExtraStrings.from_buffer(buffer)
        return self


class ResourceEditInfo2(ResourceEditInfo):
    @classmethod
    def from_buffer(cls, buffer: Buffer, resource: CompiledResource):
        vkv = BinaryKeyValues.from_buffer(buffer).root
        self = cls(buffer, resource)
        self.inputs = InputDependencies.from_vkv3(vkv['m_InputDependencies'])
        self.additional_inputs = AdditionalInputDependencies.from_vkv3(vkv['m_AdditionalInputDependencies'])
        self.arguments = ArgumentDependencies.from_vkv3(vkv['m_ArgumentDependencies'])
        self.special_deps = SpecialDependencies.from_vkv3(vkv['m_SpecialDependencies'])
        # self.custom_deps = CustomDependencies.from_vkv3(vkv)
        self.additional_files = AdditionalRelatedFiles.from_vkv3(vkv['m_AdditionalRelatedFiles'])
        self.child_resources = ChildResources.from_vkv3(vkv['m_ChildResourceList'])
        # self.extra_ints = ExtraInts.from_vkv3(vkv)
        # self.extra_floats = ExtraFloats.from_vkv3(vkv)
        # self.extra_strings = ExtraStrings.from_vkv3(vkv)
        return self
