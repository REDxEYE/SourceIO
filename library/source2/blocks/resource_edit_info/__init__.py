from dataclasses import dataclass

from SourceIO.library.utils import Buffer

from SourceIO.library.source2.blocks.base import BaseBlock
from SourceIO.library.source2.keyvalues3.binary_keyvalues import read_valve_keyvalue3, write_valve_keyvalue3
from .dependencies import *
from SourceIO.library.source2.utils.ntro_reader import NTROBuffer
from ...keyvalues3.enums import KV3Signature, KV3Format, KV3CompressionMethod
from ...keyvalues3.types import Object


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

    def to_buffer(self, buffer: Buffer):
        inputs_buffer = self.inputs.data_to_buffer()
        additional_inputs_buffer = self.additional_inputs.data_to_buffer()
        arguments_buffer = self.arguments.data_to_buffer()
        special_deps_buffer = self.special_deps.data_to_buffer()
        custom_deps_buffer = self.custom_deps.data_to_buffer()
        additional_files_buffer = self.additional_files.data_to_buffer()
        child_resources_buffer = self.child_resources.data_to_buffer()
        extra_ints_buffer = self.extra_ints.data_to_buffer()
        extra_floats_buffer = self.extra_floats.data_to_buffer()
        extra_strings_buffer = self.extra_strings.data_to_buffer()

        inputs_label = buffer.new_label("inputs", 8, None)
        additional_inputs_label = buffer.new_label("additional_inputs", 8, None)
        arguments_label = buffer.new_label("arguments", 8, None)
        special_deps_label = buffer.new_label("special_deps", 8, None)
        custom_deps_label = buffer.new_label("custom_deps", 8, None)
        additional_files_label = buffer.new_label("additional_files", 8, None)
        child_resources_label = buffer.new_label("child_resources", 8, None)
        extra_ints_label = buffer.new_label("extra_ints", 8, None)
        extra_floats_label = buffer.new_label("extra_floats", 8, None)
        extra_strings_label = buffer.new_label("extra_strings", 8, None)

        labels_and_buffers = [
            (inputs_label, inputs_buffer, self.inputs),
            (additional_inputs_label, additional_inputs_buffer, self.additional_inputs),
            (arguments_label, arguments_buffer, self.arguments),
            (special_deps_label, special_deps_buffer, self.special_deps),
            (custom_deps_label, custom_deps_buffer, self.custom_deps),
            (additional_files_label, additional_files_buffer, self.additional_files),
            (child_resources_label, child_resources_buffer, self.child_resources),
            (extra_ints_label, extra_ints_buffer, self.extra_ints),
            (extra_floats_label, extra_floats_buffer, self.extra_floats),
            (extra_strings_label, extra_strings_buffer, self.extra_strings),
        ]

        for label, buf, data in labels_and_buffers:
            data_offset = buffer.tell() - label.offset
            buffer.write(buf.data)
            label.write('2I', data_offset, len(data))


class ResourceEditInfo2(ResourceEditInfo):
    @classmethod
    def from_buffer(cls, buffer: Buffer):
        vkv = read_valve_keyvalue3(buffer)
        return cls(
            InputDependencies.from_vkv3(vkv['m_InputDependencies']),
            AdditionalInputDependencies.from_vkv3(vkv.get('m_AdditionalInputDependencies', [])),
            ArgumentDependencies.from_vkv3(vkv.get('m_ArgumentDependencies', [])),
            SpecialDependencies.from_vkv3(vkv.get('m_SpecialDependencies', [])),
            CustomDependencies(),
            AdditionalRelatedFiles.from_vkv3(vkv.get('m_AdditionalRelatedFiles', [])),
            ChildResources.from_vkv3(vkv.get('m_ChildResourceList', [])),
            ExtraInts(),
            ExtraFloats(),
            ExtraStrings()

        )

    def to_buffer(self, buffer: Buffer):
        vkv = Object({})
        if self.inputs:
            vkv["m_InputDependencies"] = self.inputs.to_vkv3()
        if self.additional_inputs:
            vkv["m_AdditionalInputDependencies"] = self.additional_inputs.to_vkv3()
        if self.arguments:
            vkv["m_ArgumentDependencies"] = self.arguments.to_vkv3()
        if self.special_deps:
            vkv["m_SpecialDependencies"] = self.special_deps.to_vkv3()
        if self.custom_deps:
            vkv["m_CustomDependencies"] = self.custom_deps.to_vkv3()
        if self.additional_files:
            vkv["m_AdditionalRelatedFiles"] = self.additional_files.to_vkv3()
        if self.child_resources:
            vkv["m_ChildResourceList"] = self.child_resources.to_vkv3()
        write_valve_keyvalue3(buffer, vkv, KV3Format.generic, KV3Signature.KV3_V3, KV3CompressionMethod.LZ4)
