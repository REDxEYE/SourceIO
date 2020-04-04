import struct
from enum import IntFlag
from typing import List

from ...byte_io_mdl import ByteIO

max_bones_per_vertex = 3
extra_8 = True


class SourceVtxFileData:
    def __init__(self):
        self.version = 0
        self.vertex_cache_size = 0
        self.max_bones_per_strip = 3
        self.max_bones_per_tri = 3
        self.max_bones_per_vertex = 3
        self.checksum = 0
        self.lod_count = 0
        self.material_replacement_list_offset = 0
        self.body_part_count = 0
        self.body_part_offset = 0

        self.vtx_body_parts = []  # type: List[SourceVtxBodyPart]
        self.material_replacement_lists = []  # type: List[MaterialReplacementList]

    def read(self, reader: ByteIO):
        self.version = reader.read_uint32()
        if self.version != 7:
            if self.version == 402653184:
                reader.insert_begin(b'\x07')
                self.version = reader.read_uint32()
                print('VTX FILE WAS "PROTECTED", but screew it :P')
            else:
                raise NotImplementedError(
                    'VTX version {} is not supported!'.format(
                        self.version))
        self.vertex_cache_size = reader.read_uint32()
        self.max_bones_per_strip = reader.read_uint16()
        self.max_bones_per_tri = reader.read_uint16()
        self.max_bones_per_vertex = reader.read_uint32()
        self.checksum = reader.read_uint32()
        self.lod_count = reader.read_uint32()
        self.material_replacement_list_offset = reader.read_uint32()
        self.body_part_count = reader.read_uint32()
        self.body_part_offset = reader.read_uint32()
        global max_bones_per_vertex
        max_bones_per_vertex = self.max_bones_per_vertex
        if self.body_part_offset > 0:

            reader.seek(self.body_part_offset)
            try:
                for _ in range(self.body_part_count):
                    self.vtx_body_parts.append(
                        SourceVtxBodyPart().read(reader))
            except struct.error:
                global extra_8
                extra_8 = False
                self.vtx_body_parts.clear()
                reader.seek(self.body_part_offset)
                for _ in range(self.body_part_count):
                    self.vtx_body_parts.append(
                        SourceVtxBodyPart().read(reader))
        if self.material_replacement_list_offset > 0:
            reader.seek(self.material_replacement_list_offset)
            for _ in range(self.lod_count):
                self.material_replacement_lists.append(
                    MaterialReplacementList().read(reader))
        # print(self.max_bones_per_vertex)

    def __repr__(self):
        return "<FileData version:{} lod count:{} body part count:{}".format(self.version,
                                                                             self.lod_count,
                                                                             self.body_part_count,
                                                                             self.vtx_body_parts)


class MaterialReplacementList:

    def __init__(self):
        self.replacements_count = 0
        self.replacement_offset = 0
        self.replacements = []  # type: List[MaterialReplacement]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.replacements_count = reader.read_int32()
        self.replacement_offset = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(entry + self.replacement_offset)
            for _ in range(self.replacements_count):
                mat = MaterialReplacement()
                mat.read(reader)
                self.replacements.append(mat)
        return self

    def __repr__(self):
        return '<MaterialReplacementList replacement count:{}>'.format(
            self.replacements_count)


class MaterialReplacement:

    def __init__(self):
        self.material_id = 0
        self.replacement_material_name_offset = 0
        self.replacement_material_name = ''

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.material_id = reader.read_int16()
        self.replacement_material_name_offset = reader.read_int32()
        self.replacement_material_name = reader.read_from_offset(entry+self.replacement_material_name_offset,
                                                                 reader.read_ascii_string)
        return self

    def __repr__(self):
        return '<MaterialReplacement mat id:{} -> "{}">'.format(
            self.material_id, self.replacement_material_name)


class SourceVtxBodyPart:
    def __init__(self):
        self.model_count = 0
        self.model_offset = 0
        self.vtx_models = []  # type: List[SourceVtxModel]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.model_count, self.model_offset = reader.read_fmt('II')

        with reader.save_current_pos():
            reader.seek(entry + self.model_offset)
            for _ in range(self.model_count):
                self.vtx_models.append(SourceVtxModel().read(reader))
        return self

    def __repr__(self):
        return "<BodyPart model_path count:{}>".format(self.model_count)


class SourceVtxModel:
    def __init__(self):
        self.lodCount = 0
        self.lodOffset = 0
        self.vtx_model_lods = []  # type: List[SourceVtxModelLod]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.lodCount, self.lodOffset = reader.read_fmt('ii')
        with reader.save_current_pos():
            if self.lodCount > 0 and self.lodOffset != 0:
                reader.seek(entry + self.lodOffset)
                for _ in range(self.lodCount):
                    self.vtx_model_lods.append(
                        SourceVtxModelLod().read(reader, self))
        return self

    def __repr__(self):
        return "<Model  lod count:{}>".format(self.lodCount)


class SourceVtxModelLod:
    def __init__(self):
        self.lod = 0
        self.meshCount = 0
        self.meshOffset = 0
        self.switchPoint = 0
        self.vtx_meshes = []  # type: List[SourceVtxMesh]

    def read(self, reader: ByteIO, model: SourceVtxModel):
        entry = reader.tell()
        self.lod = len(model.vtx_model_lods)
        self.meshCount = reader.read_uint32()
        self.meshOffset = reader.read_uint32()
        self.switchPoint = reader.read_float()
        with reader.save_current_pos():
            if self.meshOffset > 0:
                reader.seek(entry + self.meshOffset)
                for _ in range(self.meshCount):
                    self.vtx_meshes.append(SourceVtxMesh().read(reader))
        return self

    def __repr__(self):
        return "<ModelLod mesh_data count:{} switch point:{}>".format(
            self.meshCount, self.switchPoint)


class SourceVtxMesh:

    def __init__(self):
        self.strip_group_count = 0
        self.strip_group_offset = 0
        self.flags = 0
        self.vtx_strip_groups = []  # type: List[SourceVtxStripGroup]

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.strip_group_count = reader.read_uint32()
        self.strip_group_offset = reader.read_uint32()
        self.flags = reader.read_uint8()
        with reader.save_current_pos():
            if self.strip_group_offset > 0:
                reader.seek(entry + self.strip_group_offset)
                for _ in range(self.strip_group_count):
                    self.vtx_strip_groups.append(
                        SourceVtxStripGroup().read(reader))
        return self

    def __repr__(self):
        return "<Mesh strip group count:{} stripgroup offset:{}>".format(self.strip_group_count,
                                                                         self.strip_group_offset,
                                                                         )


class StripGroupFlags(IntFlag):
    STRIPGROUP_IS_FLEXED = 0x01
    STRIPGROUP_IS_HWSKINNED = 0x02
    STRIPGROUP_IS_DELTA_FLEXED = 0x04
    # NOTE: This is a temporary flag used at run time.
    STRIPGROUP_SUPPRESS_HW_MORPH = 0x08


class SourceVtxStripGroup:

    def __init__(self):
        self.vertex_count = 0
        self.vertex_offset = 0
        self.index_count = 0
        self.index_offset = 0
        self.strip_count = 0
        self.strip_offset = 0
        self.flags = 0
        self.topology_indices_count = 0
        self.topology_offset = 0
        self.vtx_vertexes = []  # type: List[SourceVtxVertex]
        self.vtx_indexes = []
        self.vtx_strips = []  # type: List[SourceVtxStrip]
        self.topology = []
        self.retry = 0

    def read(self, reader: ByteIO):

        entry = reader.tell()
        self.vertex_count = reader.read_uint32()
        self.vertex_offset = reader.read_uint32()
        self.index_count = reader.read_uint32()
        self.index_offset = reader.read_uint32()
        self.strip_count = reader.read_uint32()
        self.strip_offset = reader.read_uint32()
        self.flags = StripGroupFlags(reader.read_uint8())
        global extra_8
        if extra_8:
            self.topology_indices_count = reader.read_uint32()
            self.topology_offset = reader.read_uint32()

        with reader.save_current_pos():
            reader.seek(entry + self.index_offset)
            for _ in range(self.index_count):
                self.vtx_indexes.append(reader.read_uint16())
            reader.seek(entry + self.vertex_offset)
            for _ in range(self.vertex_count):
                SourceVtxVertex().read(reader, self)
            reader.seek(entry + self.strip_offset)
            for _ in range(self.strip_count):
                SourceVtxStrip().read(reader, self)
            if extra_8:
                reader.seek(entry + self.topology_offset)
                # for _ in range(self.topology_indices_count):
                self.topology = (
                    reader.read_bytes(
                        self.topology_indices_count * 2))

        return self

    def __repr__(self):
        return "<StripGroup Vertex count:{} Index count:{} Strip count:{} flags:{} topolygy:{} topo offset: {}-{}>".format(
            self.vertex_count, self.index_count,
            self.strip_count, self.flags, self.topology_indices_count, self.topology_offset,
            self.topology_offset + self.topology_indices_count * 2)


class SourceVtxVertex:
    def __init__(self):
        self.bone_weight_index = []
        self.bone_count = 0
        self.original_mesh_vertex_index = 0
        self.bone_id = []

    def read(self, reader: ByteIO, stripgroup: SourceVtxStripGroup):
        global max_bones_per_vertex
        self.bone_weight_index = [reader.read_uint8()
                                  for _ in range(max_bones_per_vertex)]
        self.bone_count = reader.read_uint8()
        self.original_mesh_vertex_index = reader.read_uint16()
        self.bone_id = [reader.read_uint8()
                        for _ in range(max_bones_per_vertex)]
        stripgroup.vtx_vertexes.append(self)

    def __repr__(self):
        return "<Vertex bone:{} total bone count:{}>".format(
            self.bone_id, self.bone_count)


class StripHeaderFlags(IntFlag):
    STRIP_IS_TRILIST = 0x01
    STRIP_IS_QUADLIST_REG = 0x02  # Regular
    STRIP_IS_QUADLIST_EXTRA = 0x04  # Extraordinary


class SourceVtxStrip:
    def __init__(self):
        self.index_count = 0
        self.index_mesh_index = 0
        self.vertex_count = 0
        self.vertex_mesh_index = 0
        self.bone_count = 0
        self.flags = 0
        self.bone_state_change_count = 0
        self.bone_state_change_offset = 0
        self.topology_indices_count = 0
        self.topology_offset = 0

    def read(self, reader: ByteIO, stripgroup: SourceVtxStripGroup):
        entry = reader.tell()
        self.index_count = reader.read_uint32()
        self.index_mesh_index = reader.read_uint32()
        self.vertex_count = reader.read_uint32()
        self.vertex_mesh_index = reader.read_uint32()
        self.bone_count = reader.read_uint16()
        self.flags = StripHeaderFlags(reader.read_uint8())
        self.bone_state_change_count = reader.read_uint32()
        self.bone_state_change_offset = reader.read_uint32()
        global extra_8
        if extra_8:
            self.topology_indices_count = reader.read_int32()
            self.topology_offset = reader.read_int32()
        # print('Strip end',reader.tell())

        stripgroup.vtx_strips.append(self)

    def __repr__(self):
        return '<SourceVtxStrip index count:{} vertex count:{} flags:{} topology:{}' \
               ' topo offset: {}-{} offset into indices list:{}>'.format(
            self.index_count, self.vertex_count, self.flags, self.topology_indices_count, self.topology_offset,
            self.topology_offset + self.topology_indices_count * 2,
            self.index_mesh_index)
