from typing import List

from ....byte_io_mdl import ByteIO
from ...data_structures.mdl_data import SourceMdlFileData, SourceMdlBone, SourceMdlAttachment, SourceMdlBodyPart, \
    SourceMdlTexture, SourceMdlModel, StudioHDRFlags, SourceMdlMesh, SourceBase
from ...data_structures.source_shared import SourceVector
from ...mdl.mdl_readers.mdl_v49 import SourceMdlFile49
from ....utilities.progressbar import ProgressBar


class SourceMdlFile10(SourceMdlFile49):
    def __init__(self, reader: ByteIO):
        super().__init__(reader)
        self.reader = reader
        self.file_data = SourceMdlFileDataV10()

    def read_bones(self):
        if self.file_data.bone_count > 0:
            pb = ProgressBar(
                desc='Reading bones',
                max_=self.file_data.bone_count,
                len_=20)
            self.reader.seek(self.file_data.bone_offset, 0)
            for i in range(self.file_data.bone_count):
                pb.draw()
                SourceMdlBoneV10().read(self.reader, self.file_data)
                pb.increment(1)
                pb.draw()

    def read_attachments(self):
        if self.file_data.local_attachment_count > 0:
            self.reader.seek(self.file_data.local_attachment_offset, 0)
            pb = ProgressBar(
                desc='Reading attachments',
                max_=self.file_data.local_attachment_count,
                len_=20)
            for _ in range(self.file_data.local_attachment_count):
                pb.draw()
                SourceMdlAttachmentV10().read(self.reader, self.file_data)
                pb.increment(1)

    def read_body_parts(self):
        if self.file_data.body_part_count > 0:
            self.reader.seek(self.file_data.body_part_offset)
            pb = ProgressBar(
                desc='Reading body parts',
                max_=self.file_data.body_part_count,
                len_=20)
            for _ in range(self.file_data.body_part_count):
                pb.draw()
                SourceMdlBodyPartV10().read(self.reader, self.file_data)
                pb.increment(1)

    def read_textures(self):
        if self.file_data.texture_count < 1:
            return
        self.reader.seek(self.file_data.texture_offset)
        for _ in range(self.file_data.texture_count):
            SourceMdlTextureV10().read(self.reader, self.file_data)


class SourceMdlFileDataV10(SourceMdlFileData):

    def __init__(self):
        super().__init__()
        self.texture_data_offset = 0
        self.transition_count = 0
        self.transition_offset = 0

    def read_header00(self, reader: ByteIO):
        self.id = reader.read_fourcc()
        self.version = reader.read_int32()
        self.name = reader.read_ascii_string(64)
        self.file_size = reader.read_int32()
        self.actual_file_size = reader.size()

        self.eye_position.read(reader)

        self.hull_min_position.read(reader)
        self.hull_max_position.read(reader)

        self.view_bounding_box_min_position.read(reader)
        self.view_bounding_box_max_position.read(reader)

        self.flags = StudioHDRFlags(reader.read_int32())

        self.bone_count = reader.read_int32()
        self.bone_offset = reader.read_int32()

        self.bone_controller_count = reader.read_int32()
        self.bone_controller_offset = reader.read_int32()

        self.hitbox_set_count = reader.read_int32()
        self.hitbox_set_offset = reader.read_int32()

        self.local_sequence_count = reader.read_int32()
        self.local_sequence_offset = reader.read_int32()

        self.sequence_group_count = reader.read_int32()
        self.sequence_group_offset = reader.read_int32()

        self.texture_count = reader.read_int32()
        self.texture_offset = reader.read_int32()
        self.texture_data_offset = reader.read_int32()

        self.skin_reference_count = reader.read_int32()
        self.skin_family_count = reader.read_int32()
        self.skin_family_offset = reader.read_int32()

        self.body_part_count = reader.read_int32()
        self.body_part_offset = reader.read_int32()

        self.local_attachment_count = reader.read_int32()
        self.local_attachment_offset = reader.read_int32()

        self.sound_table = reader.read_int32()
        self.sound_index = reader.read_int32()
        self.sound_groups = reader.read_int32()
        self.sound_group_offset = reader.read_int32()

        self.transition_count = reader.read_int32()
        self.transition_offset = reader.read_int32()

    def read_header01(self, reader: ByteIO):
        return

    def read_header02(self, reader: ByteIO):
        return


class SourceMdlBoneV10(SourceMdlBone):

    def read(self, reader: ByteIO, mdl: SourceMdlFileData):
        self.name = reader.read_ascii_string(32)
        self.parentBoneIndex = reader.read_int32()
        self.flags = reader.read_int32()

        self.position.read(reader)
        self.rotation.read(reader)
        self.position_scale.read(reader)
        self.rotation_scale.read(reader)
        mdl.bones.append(self)


class SourceMdlAttachmentV10(SourceMdlAttachment):

    def read(self, reader: ByteIO, mdl: SourceMdlFileData):
        self.name = reader.read_ascii_string(32)
        self.type = reader.read_int32()
        self.parent_bone = reader.read_int32()
        self.attachmentPoint.read(reader)
        for _ in range(2):
            self.vectors.append(SourceVector().read(reader))

        mdl.attachments.append(self)


class SourceMdlBodyPartV10(SourceMdlBodyPart):

    def read(self, reader: ByteIO, mdl: SourceMdlFileData):
        self.name = reader.read_ascii_string(64)
        self.model_count = reader.read_int32()
        self.base = reader.read_int32()
        self.model_offset = reader.read_int32()

        if self.model_count > 0:
            reader.seek(self.model_offset)
            for _ in range(self.model_count):
                SourceMdlModelV10().read(reader, self)

        mdl.body_parts.append(self)


class SourceMdlTextureV10(SourceMdlTexture):

    def __init__(self):
        super().__init__()
        self.width = 0
        self.height = 0
        self.data_offset = 0

    def read(self, reader: ByteIO, mdl: SourceMdlFileData):
        self.path_file_name = reader.read_ascii_string(64)
        self.flags = reader.read_int32()
        self.width = reader.read_int32()
        self.height = reader.read_int32()
        self.data_offset = reader.read_int32()

        mdl.textures.append(self)


class SourceMdlModelV10(SourceMdlModel):

    def __init__(self):
        super().__init__()

        self.vertexes = []

        self.vertex_bone_info_offset = 0
        self.vertex_bone_info = []

        self.normal_count = 0
        self.normal_bone_info_offset = 0
        self.normal_bone_info = []
        self.normal_offset = 0

        self.normals = []

        self.group_count = 0
        self.group_offset = 0

    def read(self, reader: ByteIO, body_part: SourceMdlBodyPart):
        self.name = reader.read_ascii_string(64)
        self.type = reader.read_int32()
        self.bounding_radius = reader.read_float()
        self.mesh_count = reader.read_int32()
        self.mesh_offset = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(self.mesh_offset)
            for _ in range(self.mesh_count):
                mesh = SourceMdlMeshV10()
                mesh.read(reader, self)

        self.vertex_count = reader.read_int32()
        self.vertex_bone_info_offset = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(self.vertex_bone_info_offset)
            for _ in range(self.vertex_count):
                self.vertex_bone_info.append(reader.read_uint8())
        self.vertex_offset = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(self.vertex_offset)
            for _ in range(self.vertex_count):
                self.vertexes.append(SourceVector().read(reader))

        self.normal_count = reader.read_int32()
        self.normal_bone_info_offset = reader.read_int32()

        with reader.save_current_pos():
            reader.seek(self.normal_bone_info_offset)
            for _ in range(self.normal_count):
                self.normal_bone_info.append(reader.read_uint8())

        self.normal_offset = reader.read_int32()
        with reader.save_current_pos():
            reader.seek(self.normal_offset)
            for _ in range(self.normal_count):
                self.normals.append(SourceVector().read(reader))

        self.group_count = reader.read_int32()
        self.group_offset = reader.read_int32()

        body_part.models.append(self)


class SourceMdlMeshV10(SourceMdlMesh):

    def __init__(self):
        super().__init__()
        self.face_count = 0
        self.face_offset = 0
        self.skinref = 0
        self.normal_count = 0
        self.normal_offset = 0
        self.strips_fans = SourceMeshTriangleStripOrFan10()  # type: SourceMeshTriangleStripOrFan10
        self.faces = []  # type: List[SourceMeshTriangleStripOrFan10]

    def read(self, reader: ByteIO, model: SourceMdlModel):
        self.face_count = reader.read_int32()
        self.face_offset = reader.read_int32()
        self.skinref = reader.read_int32()
        self.normal_count = reader.read_int32()
        self.normal_offset = reader.read_int32()
        self.strips_fans.read(reader, self)
        ...


class SourceMeshTriangleStripOrFan10(SourceBase):

    def __init__(self):
        self.strip = False
        self.list_count = 0
        self.vertex_infos = []

    def read(self, reader: ByteIO, mesh: SourceMdlMeshV10):
        while 1:
            self.list_count = reader.read_int16()
            if self.list_count == 0:
                break
            if self.list_count < 0:
                self.strip = False
            else:
                self.strip = True
            for _ in range(abs(self.list_count)):
                vi = SourceMdlVertexInfo10()
                vi.read(reader)
                self.vertex_infos.append(vi)

            mesh.faces.append(self)


class SourceMdlVertexInfo10(SourceBase):
    def __init__(self):
        self.vertex_index = 0
        self.normal_index = 0
        self.s = 0
        self.t = 0

    def read(self, reader: ByteIO):
        self.vertex_index = reader.read_uint16()
        self.normal_index = reader.read_uint16()
        self.s = reader.read_int16()
        self.t = reader.read_int16()
