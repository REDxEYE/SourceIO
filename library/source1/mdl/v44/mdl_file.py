import math
import traceback
from typing import List, Dict

import numpy as np

from ....utils.byte_io_mdl import ByteIO
from ....shared.base import Base

from ..v49.flex_expressions import *
from ..structs.header import MdlHeaderV44
from ..structs.bone import BoneV49
from ..structs.material import MaterialV49
from ..structs.flex import FlexController, FlexRule, FlexControllerUI, FlexOpType
from ..structs.anim_desc import AnimDesc
from ..structs.sequence import Sequence
from ..structs.attachment import AttachmentV49
from ..structs.bodygroup import BodyPartV44
from ....utils.kv_parser import ValveKeyValueParser


class _AnimBlocks:
    def __init__(self):
        self.name = ''
        self.blocks = []


class MdlV44(Base):

    def __init__(self, filepath):
        self.store_value("MDL", self)
        self.reader = ByteIO(filepath)
        self.header = MdlHeaderV44()
        self.bones = []  # type: List[BoneV49]
        self.skin_groups = []  # type: List[List[str]]
        self.materials = []  # type: List[MaterialV49]
        self.materials_paths = []

        self.flex_names = []  # type:List[str]
        self.flex_controllers = []  # type:List[FlexController]
        self.flex_ui_controllers = []  # type:List[FlexControllerUI]
        self.flex_rules = []  # type:List[FlexRule]

        self.body_parts = []  # type:List[BodyPartV44]

        self.attachments = []  # type:List[AttachmentV49]
        self.anim_descs = []  # type:List[AnimDesc]
        self.sequences = []  # type:List[Sequence]
        self.anim_block = _AnimBlocks()

        self.bone_table_by_name = []
        self.eyeballs = []

        self.key_values_raw = ''
        self.key_values = {}

    @staticmethod
    def calculate_crc(buffer):
        correct_buffer_size = math.ceil(len(buffer) / 4) * 4
        buffer += b'\x00' * (correct_buffer_size - len(buffer))

        buffer: np.ndarray = np.frombuffer(buffer, np.uint32).copy()

        orig_checksum = buffer[2]
        buffer[8 // 4] = 0
        buffer[76 // 4] = 0
        buffer[1432 // 4:1432 // 4 + 2] = 0
        buffer[1520 // 4:(1520 + 36) // 4] = 0
        buffer[1604 // 4] = 0
        with open('test.bin', 'wb') as f:
            f.write(buffer.tobytes())

        new_checksum = 0
        for i in range(buffer.shape[0]):
            tmp = buffer[i] + (new_checksum >> 27 & 1)

            new_checksum = (tmp & 0xFFFFFFFF) + ((2 * new_checksum) & 0xFFFFFFFF)
            new_checksum &= 0xFFFFFFFF
            print(f'{i * 4 + 4}: {new_checksum:08x} : {new_checksum}')
            buffer[2] = new_checksum
        print(orig_checksum, new_checksum)

    def read(self):
        reader = self.reader
        header = self.header
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
            body_part = BodyPartV44()
            body_part.read(reader)
            self.body_parts.append(body_part)

        reader.seek(header.key_value_offset)
        self.key_values_raw = reader.read(header.key_value_size).strip(b'\x00').decode('latin1')
        if self.key_values_raw:
            parser = ValveKeyValueParser(buffer_and_name=(self.key_values_raw, 'memory'), self_recover=True)
            parser.parse()
            self.key_values = parser.tree

        # self.reader.seek(self.header.local_animation_offset)
        # for _ in range(self.header.local_animation_count):
        #     anim_desc = AnimDesc()
        #     anim_desc.read(self.reader)
        #     self.anim_descs.append(anim_desc)
        #
        # self.reader.seek(self.header.local_sequence_offset)
        # for _ in range(self.header.local_sequence_count):
        #     seq = Sequence()
        #     seq.read(self.reader)
        #     self.sequences.append(seq)

        # self.anim_block.name = self.reader.read_from_offset(self.header.anim_block_name_offset,
        #                                                     self.reader.read_ascii_string)
        # self.reader.seek(self.header.anim_block_offset)
        # for _ in range(self.header.anim_block_count):
        #     self.anim_block.blocks.append(self.reader.read_fmt('2i'))
        #
        # if self.header.bone_table_by_name_offset and self.bones:
        #     self.reader.seek(self.header.bone_table_by_name_offset)
        #     self.bone_table_by_name = [self.reader.read_uint8() for _ in range(len(self.bones))]

        # for anim

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
                        inputs.append((self.flex_controllers[op.index].name, 'fetch1'))
                        fc_ui = flex_controllers[self.flex_controllers[op.index].name]
                        stack.append(FetchController(fc_ui.name, fc_ui.stereo))
                    elif flex_op == FlexOpType.FETCH2:
                        inputs.append((self.flex_names[op.index], 'fetch2'))
                        stack.append(FetchFlex(self.flex_names[op.index]))
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
                        stack.append(Combo(*[stack.pop(-1) for _ in range(op.index)]))
                    elif flex_op == FlexOpType.DOMINATE:
                        stack.append(Dominator(*[stack.pop(-1) for _ in range(op.index + 1)]))
                    elif flex_op == FlexOpType.TWO_WAY_0:
                        inputs.append((self.flex_controllers[op.index].name, '2WAY0'))
                        fc_ui = flex_controllers[self.flex_controllers[op.index].name]
                        stack.append(RClamp(FetchController(fc_ui.name, fc_ui.stereo),
                                            -1, 0, 1, 0))
                    elif flex_op == FlexOpType.TWO_WAY_1:
                        inputs.append((self.flex_controllers[op.index].name, '2WAY1'))
                        fc_ui = flex_controllers[self.flex_controllers[op.index].name]
                        stack.append(Clamp(FetchController(fc_ui.name, fc_ui.stereo), 0, 1), )
                    elif flex_op == FlexOpType.NWAY:

                        inputs.append((self.flex_controllers[op.index].name, 'NWAY'))
                        fc_ui = flex_controllers[self.flex_controllers[op.index].name]
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
                        close_lid_v_controller = self.flex_controllers[op.index]
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
                        close_lid_v_controller = self.flex_controllers[op.index]
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
