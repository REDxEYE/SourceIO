from SourceIO.library.source2.utils.ntro_reader import NTROBuffer
from .kv3_block import KVBlock


class ManifestBlock(KVBlock):

    @classmethod
    def from_buffer(cls, buffer: NTROBuffer) -> 'KVBlock':
        data = {}
        if buffer.has_ntro:
            data.update(buffer.read_struct("ResourceManifest_t"))
        version = buffer.peek_uint32()
        if version == 8:
            data["version"] = buffer.read_uint32()
            data["resources"] = []
            for _ in range(buffer.read_uint32()):
                resources = []
                start_offset = buffer.tell()
                offset = buffer.read_uint32()
                count = buffer.read_uint32()
                for _ in range(count):
                    with buffer.read_from_offset(start_offset + offset + buffer.read_uint32()):
                        resources.append(buffer.read_ascii_string())
                    offset += 4
                data["resources"].append(resources)
                buffer.skip(8)
        return cls(data)
