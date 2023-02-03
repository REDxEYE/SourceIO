import traceback
from dataclasses import dataclass
from typing import Dict, List, Mapping

import numpy.typing as npt

from ..structs.local_animation import StudioAnimDesc
from ..structs.sequence import StudioSequence
from ....utils import Buffer
from ....utils.kv_parser import ValveKeyValueParser
from .. import Mdl
from ..structs.attachment import Attachment
from ..structs.bodygroup import BodyPart
from ..structs.bone import Bone
from ..structs.flex import (FlexController, FlexControllerUI, FlexOpType,
                            FlexRule)
from ..structs.header import MdlHeaderV44
from ..structs.material import MaterialV49
from ..v49.flex_expressions import *


class _AnimBlocks:
    def __init__(self):
        self.name = ''
        self.blocks = []


@dataclass(slots=True)
class MdlV44(Mdl):
    header: MdlHeaderV44

    bones: List[Bone]
    skin_groups: List[List[str]]
    materials: List[MaterialV49]
    materials_paths: List[str]

    flex_names: List[str]
    flex_controllers: List[FlexController]
    flex_ui_controllers: List[FlexControllerUI]
    flex_rules: List[FlexRule]

    body_parts: List[BodyPart]

    attachments: List[Attachment]

    anim_descs: List[StudioAnimDesc]
    sequences: List[StudioSequence]
    animations: List[npt.NDArray]

    key_values_raw: str
    key_values: Mapping

    include_models: List[str]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = MdlHeaderV44.from_buffer(buffer)

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

        buffer.seek(header.key_value_offset)
        key_values_raw = buffer.read(header.key_value_size).strip(b'\x00').decode('latin1')
        if key_values_raw:
            parser = ValveKeyValueParser(buffer_and_name=(key_values_raw, 'memory'), self_recover=True)
            parser.parse()
            key_values = parser.tree
        else:
            key_values = {}
        # buffer.seek(header.local_animation_offset)
        # for _ in range(header.local_animation_count):
        #     anim_desc = AnimDesc()
        #     anim_desc.read(buffer)
        #     anim_descs.append(anim_desc)
        #
        # buffer.seek(header.local_sequence_offset)
        # for _ in range(header.local_sequence_count):
        #     seq = Sequence()
        #     seq.read(buffer)
        #     sequences.append(seq)

        # anim_block.name = buffer.read_from_offset(header.anim_block_name_offset,
        #                                                     buffer.read_ascii_string)
        # buffer.seek(header.anim_block_offset)
        # for _ in range(header.anim_block_count):
        #     anim_block.blocks.append(buffer.read_fmt('2i'))
        #
        # if header.bone_table_by_name_offset and bones:
        #     buffer.seek(header.bone_table_by_name_offset)
        #     bone_table_by_name = [buffer.read_uint8() for _ in range(len(bones))]

        # for anim
        return cls(header, bones, skin_groups, materials, materials_paths, flex_names, flex_controllers,
                   flex_ui_controllers, flex_rules, body_parts, attachments, key_values_raw, key_values)

    def rebuild_flex_rules(self):
        flex_controllers: Dict[str, FlexControllerUI] = {f.left_controller: f for f in self.flex_ui_controllers if
                                                         f.stereo}
        flex_controllers.update({f.right_controller: f for f in self.flex_ui_controllers if f.stereo})
        flex_controllers.update({f.nway_controller: f for f in self.flex_ui_controllers if f.nway_controller})
        flex_controllers.update({f.name: f for f in self.flex_ui_controllers})
        rules = {}
        for rule in self.flex_rules:
            stack = []
            inputs = []
            # if 1:
            try:
                for op in rule.flex_ops:
                    flex_op = op.op
                    if flex_op == FlexOpType.CONST:
                        stack.append(Value(op.value))
                    elif flex_op == FlexOpType.FETCH1:
                        inputs.append((self.flex_controllers[op.value].name, 'fetch1'))
                        fc_ui = flex_controllers[self.flex_controllers[op.value].name]
                        stack.append(FetchController(fc_ui.name, fc_ui.stereo))
                    elif flex_op == FlexOpType.FETCH2:
                        inputs.append((self.flex_names[op.value], 'fetch2'))
                        stack.append(FetchFlex(self.flex_names[op.value]))
                    elif flex_op == FlexOpType.ADD:
                        stack.append(Add(stack.pop(-1), stack.pop(-1)))
                    elif flex_op == FlexOpType.SUB:
                        stack.append(Sub(stack.pop(-1), stack.pop(-1)))
                    elif flex_op == FlexOpType.MUL:
                        stack.append(Mul(stack.pop(-1), stack.pop(-1)))
                    elif flex_op == FlexOpType.DIV:
                        stack.append(Div(stack.pop(-1), stack.pop(-1)))
                    elif flex_op == FlexOpType.NEG:
                        stack.append(Neg(stack.pop(-1)))
                    elif flex_op == FlexOpType.MAX:
                        stack.append(Max(stack.pop(-1), stack.pop(-1)))
                    elif flex_op == FlexOpType.MIN:
                        stack.append(Min(stack.pop(-1), stack.pop(-1)))
                    elif flex_op == FlexOpType.COMBO:
                        stack.append(Combo(*[stack.pop(-1) for _ in range(op.value)]))
                    elif flex_op == FlexOpType.DOMINATE:
                        stack.append(Dominator(*[stack.pop(-1) for _ in range(op.value + 1)]))
                    elif flex_op == FlexOpType.TWO_WAY_0:
                        inputs.append((self.flex_controllers[op.value].name, '2WAY0'))
                        fc_ui = flex_controllers[self.flex_controllers[op.value].name]
                        stack.append(RClamp(FetchController(fc_ui.name, fc_ui.stereo),
                                            -1, 0, 1, 0))
                    elif flex_op == FlexOpType.TWO_WAY_1:
                        inputs.append((self.flex_controllers[op.value].name, '2WAY1'))
                        fc_ui = flex_controllers[self.flex_controllers[op.value].name]
                        stack.append(Clamp(FetchController(fc_ui.name, fc_ui.stereo), 0, 1), )
                    elif flex_op == FlexOpType.NWAY:

                        inputs.append((self.flex_controllers[op.value].name, 'NWAY'))
                        fc_ui = flex_controllers[self.flex_controllers[op.value].name]
                        flex_cnt = FetchController(fc_ui.name, fc_ui.stereo)

                        flex_cnt_value = int(stack.pop(-1).value)
                        inputs.append((self.flex_controllers[flex_cnt_value].name, 'NWAY'))
                        fc_ui = flex_controllers[self.flex_controllers[flex_cnt_value].name]
                        multi_cnt = FetchController(fc_ui.nway_controller, fc_ui.stereo)

                        # Reversed the order, revert back if it wont help
                        f_w = stack.pop(-1)
                        f_z = stack.pop(-1)
                        f_y = stack.pop(-1)
                        f_x = stack.pop(-1)
                        final_expr = NWay(multi_cnt, flex_cnt, f_x, f_y, f_z, f_w)
                        stack.append(final_expr)
                    elif flex_op == FlexOpType.DME_UPPER_EYELID:
                        close_lid_v_controller = self.flex_controllers[op.value]
                        inputs.append((close_lid_v_controller.name, 'DUE'))
                        close_lid_v = RClamp(FetchController(close_lid_v_controller.name),
                                             close_lid_v_controller.min, close_lid_v_controller.max,
                                             0, 1)

                        flex_cnt_value = int(stack.pop(-1).value)
                        close_lid_controller = self.flex_controllers[flex_cnt_value]
                        inputs.append((close_lid_controller.name, 'DUE'))
                        close_lid = RClamp(FetchController(close_lid_controller.name),
                                           close_lid_controller.min, close_lid_controller.max,
                                           0, 1)

                        blink_index = int(stack.pop(-1).value)
                        # blink = Value(0.0)
                        # if blink_index >= 0:
                        #     blink_controller = self.flex_controllers[blink_index]
                        #     inputs.append((blink_controller.name, 'DUE'))
                        #     blink_fetch = FetchController(blink_controller.name)
                        #     blink = CustomFunction('rclamped', blink_fetch,
                        #                            blink_controller.min, blink_controller.max,
                        #                            0, 1)

                        eye_up_down_index = int(stack.pop(-1).value)
                        eye_up_down = Value(0.0)
                        if eye_up_down_index >= 0:
                            eye_up_down_controller = self.flex_controllers[eye_up_down_index]
                            inputs.append((eye_up_down_controller.name, 'DUE'))
                            eye_up_down_fetch = FetchController(eye_up_down_controller.name)
                            eye_up_down = RClamp(eye_up_down_fetch,
                                                 eye_up_down_controller.min, eye_up_down_controller.max,
                                                 -1, 1)

                        stack.append(CustomFunction('upper_eyelid_case', eye_up_down, close_lid_v, close_lid))
                    elif flex_op == FlexOpType.DME_LOWER_EYELID:
                        close_lid_v_controller = self.flex_controllers[op.value]
                        inputs.append((close_lid_v_controller.name, 'DUE'))
                        close_lid_v = RClamp(FetchController(close_lid_v_controller.name),
                                             close_lid_v_controller.min, close_lid_v_controller.max,
                                             0, 1)

                        flex_cnt_value = int(stack.pop(-1).value)
                        close_lid_controller = self.flex_controllers[flex_cnt_value]
                        inputs.append((close_lid_controller.name, 'DUE'))
                        close_lid = RClamp(FetchController(close_lid_controller.name),
                                           close_lid_controller.min, close_lid_controller.max,
                                           0, 1)

                        blink_index = int(stack.pop(-1).value)
                        # blink = Value(0.0)
                        # if blink_index >= 0:
                        #     blink_controller = self.flex_controllers[blink_index]
                        #     inputs.append((blink_controller.name, 'DUE'))
                        #     blink_fetch = FetchController(blink_controller.name)
                        #     blink = CustomFunction('rclamped', blink_fetch,
                        #                            blink_controller.min, blink_controller.max,
                        #                            0, 1)

                        eye_up_down_index = int(stack.pop(-1).value)
                        eye_up_down = Value(0.0)
                        if eye_up_down_index >= 0:
                            eye_up_down_controller = self.flex_controllers[eye_up_down_index]
                            inputs.append((eye_up_down_controller.name, 'DUE'))
                            eye_up_down_fetch = FetchController(eye_up_down_controller.name)
                            eye_up_down = RClamp(eye_up_down_fetch,
                                                 eye_up_down_controller.min, eye_up_down_controller.max,
                                                 -1, 1)

                        stack.append(CustomFunction('lower_eyelid_case', eye_up_down, close_lid_v, close_lid))
                    elif flex_op == FlexOpType.OPEN:
                        continue
                    else:
                        print("Unknown OP", op)
                if len(stack) > 1 or not stack:
                    print(f"failed to parse ({self.flex_names[rule.flex_index]}) flex rule")
                    print(stack)
                    continue
                final_expr = stack.pop(-1)
                name = self.flex_names[rule.flex_index]
                rules[name] = (final_expr, inputs)
            except Exception as ex:
                traceback.print_exc()
                print(f"failed to parse ({self.flex_names[rule.flex_index]}) flex rule")
                print(stack)

        return rules
