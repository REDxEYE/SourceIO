from pathlib import Path

from ..utilities.hexify import rhex
from ..byte_io_mdl import ByteIO


class CompiledShader:
    MAGIC = "vcs2"

    def __init__(self, path: str):
        self.path = Path(path)
        self.reader = ByteIO(path=self.path)
        self.shader_type = "UNKNOWN"
        self.version = 0
        self.file_id = b''
        self.static_id = b''
        self.shader_parameters = []

    def read(self):
        if self.path.stem.endswith('vs'):
            self.shader_type = 'vertex'
        elif self.path.stem.endswith('ps'):
            self.shader_type = 'pixel'
        elif self.path.stem.endswith('features'):
            self.shader_type = 'features'
        else:
            raise NotImplementedError(f"Unknown shader type: {self.path.stem}")
        reader = self.reader
        magic = reader.read_fourcc()
        assert magic == self.MAGIC, f"Invalid vcs magic ({magic}!={self.MAGIC})"
        self.version = reader.read_uint32()
        zero1 = reader.read_uint32()
        assert zero1 == 0, "non zero value on 0x8 offset"
        if self.shader_type == "features":
            print("skip features for now")
            return
        else:
            self.read_shader()

    def read_shader(self):
        reader = self.reader
        self.file_id = reader.read_bytes(16)
        self.static_id = reader.read_bytes(16)
        print(rhex(self.file_id))
        print(rhex(self.static_id))
        print("unk:", reader.read_uint32())

        count = reader.read_uint32()

        for _ in range(count):
            name = reader.read_ascii_string(128)
            print(name)
            self.shader_parameters.append(name)
            unk = reader.read_fmt('6i')
            print("shader params:",unk)

        count = reader.read_uint32()

        for _ in range(count):
            unk = reader.read_fmt('8i')#8
            print("unk2",unk)
            reader.skip(440)
