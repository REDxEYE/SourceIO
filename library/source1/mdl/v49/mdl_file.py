from ..v44.mdl_file import MdlV44

from ..structs.header import MdlHeaderV49
from ..structs.bone import BoneV49
from ..structs.material import MaterialV49
from ..structs.flex import FlexController, FlexRule, FlexControllerUI, FlexOpType
from ..structs.anim_desc import AnimDesc
from ..structs.sequence import Sequence
from ..structs.attachment import AttachmentV49
from ..structs.bodygroup import BodyPartV49
from ....utils.kv_parser import ValveKeyValueParser


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
            parser = ValveKeyValueParser(buffer_and_name=(self.key_values_raw, 'memory'), self_recover=True)
            parser.parse()
            self.key_values = parser.tree

        try:
            reader.seek(header.local_animation_offset)
            for _ in range(header.local_animation_count):
                anim_desc = AnimDesc()
                anim_desc.read(reader)
                self.anim_descs.append(anim_desc)

            reader.seek(header.local_sequence_offset)
            for _ in range(header.local_sequence_count):
                seq = Sequence()
                seq.read(reader)
                self.sequences.append(seq)

            self.anim_block.name = reader.read_from_offset(header.anim_block_name_offset, reader.read_ascii_string)
            self.reader.seek(self.header.anim_block_offset)
            for _ in range(self.header.anim_block_count):
                self.anim_block.blocks.append(self.reader.read_fmt('2i'))

            if self.header.bone_table_by_name_offset and self.bones:
                self.reader.seek(self.header.bone_table_by_name_offset)
                self.bone_table_by_name = [self.reader.read_uint8() for _ in range(len(self.bones))]
        except:  # I don't wanna deal with animations
            pass
        # for anim
