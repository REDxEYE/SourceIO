
from SourceIO.library.source2.blocks.base import BaseBlock
from SourceIO.library.source2.blocks.resource_introspection_manifest.types import Struct, Enum
from SourceIO.library.source2.utils.ntro_reader import NTROBuffer, ResourceIntrospectionInfo


class ResourceIntrospectionManifest(BaseBlock):
    def __init__(self, info: ResourceIntrospectionInfo):
        self.info = info

    @classmethod
    def from_buffer(cls, buffer: NTROBuffer):
        version = buffer.read_uint32()
        assert version == 4, f'Introspection version {version} is not supported'
        struct_offset = buffer.read_relative_offset32()
        struct_count = buffer.read_uint32()
        enum_offset = buffer.read_relative_offset32()
        enum_count = buffer.read_uint32()

        struct_lookup = {}
        enum_lookup = {}
        structs = []
        enums = []
        with buffer.read_from_offset(struct_offset):
            for i in range(struct_count):
                struct_type = Struct.from_buffer(buffer)
                struct_lookup[struct_type.name] = struct_type
                struct_lookup[struct_type.id] = struct_type
                structs.append(struct_type)
        with buffer.read_from_offset(enum_offset):
            for i in range(enum_count):
                enum_type = Enum.from_buffer(buffer)
                enum_lookup[enum_type.name] = enum_type
                enum_lookup[enum_type.id] = enum_type
                enums.append(enum_type)
        return cls(buffer, ResourceIntrospectionInfo(version, structs, enums, struct_lookup, enum_lookup, {}))


