from typing import Optional

from . import Base
from . import ByteIO
from .material import MaterialV36, MaterialV49


class EyeballV36(Base):
    def __init__(self):
        self.name = ''
        self.bone_index = 0
        self.org = []
        self.z_offset = 0.0
        self.radius = 0.0
        self.up = []
        self.forward = []
        self.material_id = 0
        self.material: Optional[MaterialV36] = None

        self.iris_scale = 0.0
        self.upper_flex_desc = []
        self.lower_flex_desc = []
        self.upper_target = []
        self.lower_target = []

        self.upper_lid_flex_desc = 0
        self.lower_lid_flex_desc = 0
        self.eyeball_is_non_facs = 0

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name = reader.read_source1_string(entry)
        self.bone_index = reader.read_uint32()
        self.org = reader.read_fmt("3f")
        self.z_offset = reader.read_float()
        self.radius = reader.read_float()
        self.up = reader.read_fmt("3f")
        self.forward = reader.read_fmt("3f")
        self.material_id = reader.read_int32()
        reader.read_uint32()
        self.iris_scale = reader.read_float()
        reader.read_uint32()
        self.upper_flex_desc = reader.read_fmt("3I")
        self.lower_flex_desc = reader.read_fmt("3I")
        self.upper_target = reader.read_fmt("3f")
        self.lower_target = reader.read_fmt("3f")
        self.upper_lid_flex_desc = reader.read_uint32()
        self.lower_lid_flex_desc = reader.read_uint32()
        reader.skip(4 * 4)


class EyeballV44(EyeballV36):
    def __init__(self):
        super().__init__()
        self.material: Optional[MaterialV49] = None

    def read(self, reader: ByteIO):
        super().read(reader)
        reader.skip(4 * 4)
        self.eyeball_is_non_facs = reader.read_uint8()
        reader.skip(3)
        reader.skip(3 * 4)
        pass


class EyeballV49(EyeballV44):
    def __init__(self):
        super().__init__()
        self.material: Optional[MaterialV49] = None

    def read(self, reader: ByteIO):
        super().read(reader)
