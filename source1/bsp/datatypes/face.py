from ....byte_io_mdl import ByteIO


class Face:
    def __init__(self):
        self.plane_num = 0
        self.side = 0
        self.on_node = 0
        self.first_edge = 0
        self.edge_count = 0
        self.tex_info_id = 0
        self.disp_info_id = 0
        self.surface_fog_volume_id = 0
        self.styles = []
        self.light_offset = 0
        self.area = 0.0
        self.lightmap_texture_mins_in_luxels = []
        self.lightmap_texture_size_in_luxels = []
        self.orig_face = 0
        self.prim_count = 0
        self.first_prim_id = 0
        self.smoothing_groups = 0

    def parse(self, reader: ByteIO):
        self.plane_num = reader.read_uint16()
        self.side = reader.read_uint8()
        self.on_node = reader.read_uint8()
        self.first_edge = reader.read_int32()
        self.edge_count = reader.read_int16()
        self.tex_info_id = reader.read_int16()
        self.disp_info_id = reader.read_int16()
        self.surface_fog_volume_id = reader.read_int16()
        self.styles = [reader.read_int8() for _ in range(4)]
        self.light_offset = reader.read_int32()
        self.area = reader.read_float()
        self.lightmap_texture_mins_in_luxels = [reader.read_int32() for _ in range(2)]
        self.lightmap_texture_size_in_luxels = [reader.read_int32() for _ in range(2)]
        self.orig_face = reader.read_int32()
        self.prim_count = reader.read_uint16()
        self.first_prim_id = reader.read_uint16()
        self.smoothing_groups = reader.read_uint32()
        return self
