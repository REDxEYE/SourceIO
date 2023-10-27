import math
import struct
import traceback
from dataclasses import dataclass

from ..structs.local_animation import StudioAnimDesc
from ..structs.sequence import StudioSequence
from .....logger import SLoggingManager
from ....utils import Buffer
from ....utils.kv_parser import KVParserException, ValveKeyValueParser
from ..structs.attachment import Attachment
from ..structs.bodygroup import BodyPart
from ..structs.bone import Bone
from ..structs.flex import FlexController, FlexControllerUI, FlexRule
from ..structs.header import MdlHeaderV49
from ..structs.material import MaterialV49
from ..v44.mdl_file import MdlV44

log_manager = SLoggingManager()
logger = log_manager.get_logger('MDL49')


@dataclass(slots=True)
class MdlV49(MdlV44):
    header: MdlHeaderV49

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = MdlHeaderV49.from_buffer(buffer)

        bones = []
        buffer.seek(header.bone_offset)
        for bone_id in range(header.bone_count):
            bone = Bone.from_buffer(buffer, header.version)
            bone.bone_id = bone_id
            bones.append(bone)

        materials = []
        buffer.seek(header.texture_offset)
        for _ in range(header.texture_count):
            texture = MaterialV49.from_buffer(buffer, header.version)
            materials.append(texture)

        materials_paths = []
        buffer.seek(header.texture_path_offset)
        for _ in range(header.texture_path_count):
            materials_paths.append(buffer.read_source1_string(0))

        skin_groups = []
        buffer.seek(header.skin_family_offset)
        for _ in range(header.skin_family_count):
            skin_group = []
            for _ in range(header.skin_reference_count):
                texture_index = buffer.read_uint16()
                skin_group.append(materials[texture_index].name)
            skin_groups.append(skin_group)

        diff_start = 0
        for skin_info in skin_groups[1:]:
            for n, (a, b) in enumerate(zip(skin_groups[0], skin_info)):
                if a == b:
                    diff_start = max(n, diff_start)
                    break

        for n, skin_info in enumerate(skin_groups):
            skin_groups[n] = skin_info[:diff_start]

        flex_names = []
        buffer.seek(header.flex_desc_offset)
        for _ in range(header.flex_desc_count):
            flex_names.append(buffer.read_source1_string(buffer.tell()))

        flex_controllers = []
        buffer.seek(header.flex_controller_offset)
        for _ in range(header.flex_controller_count):
            controller = FlexController.from_buffer(buffer, header.version)
            flex_controllers.append(controller)

        flex_rules = []
        buffer.seek(header.flex_rule_offset)
        for _ in range(header.flex_rule_count):
            rule = FlexRule.from_buffer(buffer, header.version)
            flex_rules.append(rule)

        attachments = []
        buffer.seek(header.local_attachment_offset)
        for _ in range(header.local_attachment_count):
            attachment = Attachment.from_buffer(buffer, header.version)
            attachments.append(attachment)

        flex_ui_controllers = []
        buffer.seek(header.flex_controller_ui_offset)
        for _ in range(header.flex_controller_ui_count):
            flex_controller = FlexControllerUI.from_buffer(buffer, header.version)
            flex_ui_controllers.append(flex_controller)

        body_parts = []
        buffer.seek(header.body_part_offset)
        for _ in range(header.body_part_count):
            body_part = BodyPart.from_buffer(buffer, header.version)
            body_parts.append(body_part)

        local_animations = []
        buffer.seek(header.local_animation_offset)
        for _ in range(header.local_animation_count):
            local_animations.append(StudioAnimDesc.from_buffer(buffer))

        local_sequences = []
        buffer.seek(header.local_sequence_offset)
        for _ in range(header.local_sequence_count):
            local_sequences.append(StudioSequence.from_buffer(buffer, header.version))

        animations = []
        for anim_desc in local_animations:
            try:
                animations.append(anim_desc.read_animations(buffer, bones))
            except (AssertionError, ValueError, struct.error):
                traceback.print_exc()
                animations.extend([None] * (len(animations) - len(local_animations)))
                break

        include_models = []
        buffer.seek(header.include_model_offset)
        for inc_model in range(header.include_model_count):
            entry = buffer.tell()
            label = buffer.read_source1_string(entry)
            path = buffer.read_source1_string(entry)
            include_models.append(path)

        buffer.seek(header.key_value_offset)
        key_values_raw = buffer.read(header.key_value_size).strip(b'\x00').decode('latin1')
        key_values = {}
        if key_values_raw:
            try:
                parser = ValveKeyValueParser(buffer_and_name=(key_values_raw, 'memory'), self_recover=True,
                                             array_of_blocks=True)
                parser.parse()
                key_values = parser.tree[0]
            except KVParserException as e:
                logger.exception('Failed to parse key values due to', e)

        return cls(header, bones, skin_groups, materials, materials_paths, flex_names, flex_controllers,
                   flex_ui_controllers, flex_rules, body_parts, attachments, local_animations, local_sequences,
                   animations, key_values_raw, key_values, include_models)
