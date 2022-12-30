from .....logger import SLoggingManager
from ....utils.byte_io_mdl import ByteIO
from ....utils.kv_parser import KVParserException, ValveKeyValueParser
from ..structs.attachment import AttachmentV49
from ..structs.bodygroup import BodyPartV49
from ..structs.bone import BoneV49
from ..structs.flex import (FlexController, FlexControllerUI, FlexOpType,
                            FlexRule)
from ..structs.header import MdlHeaderV49
from ..structs.material import MaterialV49
from ..v44.mdl_file import MdlV44

log_manager = SLoggingManager()
logger = log_manager.get_logger('MDL49')


class _AnimBlocks:
    def __init__(self):
        self.name = ''
        self.blocks = []


class MdlV49(MdlV44):

    def __init__(self, filepath):
        super().__init__(filepath)
        self.store_value("MDL", self)
        self.header = MdlHeaderV49()

    def read(self):
        header = self.header
        reader = self.reader
        header.read(reader)

        reader.seek(header.bone_offset)
        for bone_id in range(header.bone_count):
            bone = BoneV49(bone_id)
            bone.read(reader)
            self.bones.append(bone)

        reader.seek(header.texture_offset)
        for _ in range(header.texture_count):
            texture = MaterialV49()
            texture.read(reader)
            self.materials.append(texture)

        reader.seek(header.texture_path_offset)
        for _ in range(header.texture_path_count):
            self.materials_paths.append(reader.read_source1_string(0))

        reader.seek(header.skin_family_offset)
        for _ in range(header.skin_family_count):
            skin_group = []
            for _ in range(header.skin_reference_count):
                texture_index = reader.read_uint16()
                skin_group.append(self.materials[texture_index].name)
            self.skin_groups.append(skin_group)

        diff_start = 0
        for skin_info in self.skin_groups[1:]:
            for n, (a, b) in enumerate(zip(self.skin_groups[0], skin_info)):
                if a == b:
                    diff_start = max(n, diff_start)
                    break

        for n, skin_info in enumerate(self.skin_groups):
            self.skin_groups[n] = skin_info[:diff_start]

        reader.seek(header.flex_desc_offset)
        for _ in range(header.flex_desc_count):
            self.flex_names.append(reader.read_source1_string(reader.tell()))

        reader.seek(header.flex_controller_offset)
        for _ in range(header.flex_controller_count):
            controller = FlexController()
            controller.read(reader)
            self.flex_controllers.append(controller)

        reader.seek(header.flex_rule_offset)
        for _ in range(header.flex_rule_count):
            rule = FlexRule()
            rule.read(reader)
            self.flex_rules.append(rule)

        reader.seek(header.local_attachment_offset)
        for _ in range(header.local_attachment_count):
            attachment = AttachmentV49()
            attachment.read(reader)
            self.attachments.append(attachment)

        reader.seek(header.flex_controller_ui_offset)
        for _ in range(header.flex_controller_ui_count):
            flex_controller = FlexControllerUI()
            flex_controller.read(reader)
            self.flex_ui_controllers.append(flex_controller)

        reader.seek(header.body_part_offset)
        for _ in range(header.body_part_count):
            body_part = BodyPartV49()
            body_part.read(reader)
            self.body_parts.append(body_part)
        reader.seek(header.key_value_offset)
        self.key_values_raw = reader.read(header.key_value_size).strip(b'\x00').decode('latin1')
        if self.key_values_raw:
            try:
                parser = ValveKeyValueParser(buffer_and_name=(self.key_values_raw, 'memory'), self_recover=True,
                                             array_of_blocks=True)
                parser.parse()
                self.key_values = parser.tree[0]
            except KVParserException as e:
                logger.exception('Failed to parse key values due to', e)
