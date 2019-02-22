from SourceIO.byte_io_mdl import ByteIO
from SourceIO.data_structures.mdl_data import SourceMdlFileData, SourceMdlBone, SourceMdlAttachment
from SourceIO.data_structures.source_shared import SourceVector
from SourceIO.mdl_readers.mdl_v49 import SourceMdlFile49
from SourceIO.utilities.progressbar import ProgressBar


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

        self.flags = reader.read_int32()

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
        self.positionScale.read(reader)
        self.rotationScale.read(reader)
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

