from .section import Section
from ...new_shared.base import Base
from ....byte_io_mdl import ByteIO


class CompactSurface(Base):

    def __init__(self):
        self.size = 0
        self.vphys_id = 0
        self.version = 0
        self.model_type = 0
        self.surface_size = 0
        self.drag_axis_area = [0, 0, 0]
        self.axis_map_size = 0
        self.unk_data = []
        self.sections = []

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.size = reader.read_uint32()
        self.vphys_id = reader.read_fourcc()

        self.version = reader.read_uint16()
        self.model_type = reader.read_uint16()

        self.surface_size = reader.read_uint32()
        self.drag_axis_area = reader.read_fmt('3f')
        self.axis_map_size = reader.read_uint32()
        self.unk_data = reader.read_fmt('12I')

        assert self.unk_data[11] == 1397773897
        end = entry + self.size + 4
        vertex_start = end
        while reader.tell() < vertex_start:
            section = Section()
            tmp = reader.tell()
            section.read(reader)
            vertex_start = tmp + section.vertex_data_offset
            self.sections.append(section)
        reader.seek(entry+self.size+4)

    # struct compactsurfaceheader_t
#  {
# 	int	size;			// Size of the content after this byte
# 	int	vphysicsID;		// Generally the ASCII for "VPHY" in newer files
# 	short	version;
# 	short	modelType;
# 	int	surfaceSize;
# 	Vector	dragAxisAreas;
# 	int	axisMapSize;
#   int dummy[12];
#  };
