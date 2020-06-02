from typing import List

from ...byte_io_mdl import ByteIO
from SourceIO.source1.new_shared.base import Base

from .structs.header import Header
from .structs.bone import Bone
from .structs.texture import Material
from .structs.flex import FlexController, FlexRule, FlexControllerUI
from .structs.attachment import Attachment
from .structs.bodygroup import BodyPart


class Mdl(Base):

    def __init__(self, filepath):
        self.reader = ByteIO(path=filepath)
        self.header = Header()
        self.bones = []  # type: List[Bone]
        self.skin_groups = []  # type: List[List[str]]
        self.materials = []  # type: List[Material]
        self.materials_paths = []

        self.flex_names = []  # type:List[str]
        self.flex_controllers = []  # type:List[FlexController]
        self.flex_ui_controllers = []  # type:List[FlexControllerUI]
        self.flex_rules = []  # type:List[FlexRule]

        self.body_parts = []  # type:List[BodyPart]

        self.attachments = []  # type:List[Attachment]

    def read(self):
        self.header.read(self.reader)

        self.reader.seek(self.header.bone_offset)
        for _ in range(self.header.bone_count):
            bone = Bone()
            bone.read(self.reader)
            self.bones.append(bone)

        self.reader.seek(self.header.texture_offset)
        for _ in range(self.header.texture_count):
            texture = Material()
            texture.read(self.reader)
            self.materials.append(texture)

        self.reader.seek(self.header.texture_path_offset)
        for _ in range(self.header.texture_path_count):
            self.materials_paths.append(self.reader.read_source1_string(0))

        self.reader.seek(self.header.skin_family_offset)
        for _ in range(self.header.skin_family_count):
            skin_group = []
            for _ in range(self.header.skin_reference_count):
                texture_index = self.reader.read_uint16()
                skin_group.append(self.materials[texture_index].name)
            self.skin_groups.append(skin_group)

        self.reader.seek(self.header.flex_desc_offset)
        for _ in range(self.header.flex_desc_count):
            self.flex_names.append(self.reader.read_source1_string(self.reader.tell()))

        self.reader.seek(self.header.flex_controller_offset)
        for _ in range(self.header.flex_controller_count):
            controller = FlexController()
            controller.read(self.reader)
            self.flex_controllers.append(controller)

        self.reader.seek(self.header.flex_rule_offset)
        for _ in range(self.header.flex_rule_count):
            rule = FlexRule()
            rule.read(self.reader)
            self.flex_rules.append(rule)

        self.reader.seek(self.header.local_attachment_offset)
        for _ in range(self.header.local_attachment_count):
            attachment = Attachment()
            attachment.read(self.reader)
            self.attachments.append(attachment)

        self.reader.seek(self.header.flex_controller_ui_offset)
        for _ in range(self.header.flex_controller_ui_count):
            flex_controller = FlexControllerUI()
            flex_controller.read(self.reader)
            self.flex_ui_controllers.append(flex_controller)

        self.reader.seek(self.header.body_part_offset)
        for _ in range(self.header.body_part_count):
            body_part = BodyPart()
            body_part.read(self.reader)
            self.body_parts.append(body_part)
