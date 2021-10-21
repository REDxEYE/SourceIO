from ...data_blocks import DATA
from ...resource_types import ValveCompiledResource


class ResourceManifest(DATA):
    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)

    @property
    def ntro_struct_name(self):
        return "ResourceManifest_t"

    def read(self):
        if self.has_ntro:
            return super().read()
        reader = self.reader
        version = reader.peek_uint32()
        if version == 8:
            self.data['version'] = reader.read_uint32()
            self.data['resources'] = []
            for _ in range(reader.read_uint32()):
                resources = []
                entry = reader.tell()
                offset = reader.read_uint32()
                count = reader.read_uint32()
                for _ in range(count):
                    string_offset = offset + reader.read_uint32()
                    with reader.save_current_pos():
                        reader.seek(entry + string_offset)
                        string = reader.read_ascii_string()
                        resources.append(string)
                    offset += 4
                self.data['resources'].append(resources)
                reader.skip(8)
        else:
            raise NotImplementedError(f'Unsupported version of resource manifest {version}')


class ValveCompiledResourceManifest(ValveCompiledResource):
    data_block_class = ResourceManifest

    def __init__(self, path_or_file):
        super().__init__(path_or_file)
