from SourceIO.library.utils import Buffer

from .kv3_block import KVBlock
from .resource_introspection_manifest import ResourceIntrospectionManifest


class ManifestBlock(KVBlock):
    # @staticmethod
    # def _get_struct(ntro):
    #     return ntro.struct_by_name("ResourceManifest_t")

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'KVBlock':
        self: 'KVBlock' = cls(buffer)
        # if self.has_ntro:
        #     ntro = self._resource.get_block(ResourceIntrospectionManifest,block_name='NTRO')
        #     self.update(ntro.read_struct(buffer, self._get_struct(ntro)))
        version = buffer.peek_uint32()
        if version == 8:
            self["version"] = buffer.read_uint32()
            self["resources"] = []
            for _ in range(buffer.read_uint32()):
                resources = []
                start_offset = buffer.tell()
                offset = buffer.read_uint32()
                count = buffer.read_uint32()
                for _ in range(count):
                    with buffer.read_from_offset(start_offset + offset + buffer.read_uint32()):
                        resources.append(buffer.read_ascii_string())
                    offset += 4
                self["resources"].append(resources)
                buffer.skip(8)
        return self