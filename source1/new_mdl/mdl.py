import traceback
from typing import List

from ...byte_io_mdl import ByteIO
from ..new_shared.base import Base

from .flex_expressions import *
from .structs.header import Header
from .structs.bone import Bone
from .structs.texture import Material
from .structs.flex import FlexController, FlexRule, FlexControllerUI, FlexOpType
from .structs.anim_desc import AnimDesc
from .structs.sequence import Sequence
from .structs.attachment import Attachment
from .structs.bodygroup import BodyPart


class _AnimBlocks:
    def __init__(self):
        self.name = ''
        self.blocks = []


class Mdl(Base):

    def __init__(self, filepath):
        self.store_value("MDL", self)
        self.reader = ByteIO(filepath)
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
        self.anim_descs = []  # type:List[AnimDesc]
        self.sequences = []  # type:List[Sequence]
        self.anim_block = _AnimBlocks()

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
        #
        # self.anim_block.name = self.reader.read_from_offset(self.header.anim_block_name_offset,
        #                                                     self.reader.read_ascii_string)
        # self.reader.seek(self.header.anim_block_offset)
        # for _ in range(self.header.anim_block_count):
        #     self.anim_block.blocks.append(self.reader.read_fmt('2i'))

    def rebuild_flex_rules(self):
        rules = {}

        for rule in self.flex_rules:
            stack = []
            try:
                for op in rule.flex_ops:
                    flex_op = op.op
                    if flex_op == FlexOpType.CONST:
                        stack.append(Value(op.value))
                    elif flex_op == FlexOpType.FETCH1:
                        stack.append(FetchController(self.flex_controllers[op.index].name))
                    elif flex_op == FlexOpType.FETCH2:
                        stack.append(FetchFlex(self.flex_names[op.index]))
                    elif flex_op == FlexOpType.ADD:
                        right = stack.pop(-1)
                        left = stack.pop(-1)
                        stack.append(Add(left, right))
                    elif flex_op == FlexOpType.SUB:
                        right = stack.pop(-1)
                        left = stack.pop(-1)
                        stack.append(Sub(left, right))
                    elif flex_op == FlexOpType.MUL:
                        right = stack.pop(-1)
                        left = stack.pop(-1)
                        stack.append(Mul(left, right))
                    elif flex_op == FlexOpType.DIV:
                        right = stack.pop(-1)
                        left = stack.pop(-1)
                        stack.append(Div(left, right))
                    elif flex_op == FlexOpType.NEG:
                        stack.append(Neg(stack.pop(-1)))
                    elif flex_op == FlexOpType.MAX:
                        right = stack.pop(-1)
                        left = stack.pop(-1)
                        stack.append(Max(left, right))
                    elif flex_op == FlexOpType.MIN:
                        right = stack.pop(-1)
                        left = stack.pop(-1)
                        stack.append(Min(left, right))
                    elif flex_op == FlexOpType.COMBO:
                        count = op.index
                        values = [stack.pop(-1) for _ in range(count)]
                        combo = Combo(*values)
                        stack.append(combo)
                    elif flex_op == FlexOpType.DOMINATE:
                        count = op.index+1
                        values = [stack.pop(-1) for _ in range(count)]
                        dom = Dominator(*values)
                        stack.append(dom)
                    elif flex_op == FlexOpType.TWO_WAY_0:
                        mx = Max(Add(FetchController(self.flex_controllers[op.index].name), Value(1.0)), Value(0.0))
                        mn = Min(mx, Value(1.0))
                        res = Sub(1, mn)
                        stack.append(res)
                    elif flex_op == FlexOpType.TWO_WAY_1:
                        mx = Max(FetchController(self.flex_controllers[op.index].name), Value(0.0))
                        mn = Min(mx, Value(1.0))
                        stack.append(mn)
                    elif flex_op == FlexOpType.NWAY:
                        flex_cnt_value = int(stack.pop(-1).value)
                        flex_cnt = FetchController(self.flex_controllers[flex_cnt_value].name)
                        f_w = stack.pop(-1)
                        f_z = stack.pop(-1)
                        f_y = stack.pop(-1)
                        f_x = stack.pop(-1)
                        gtx = Min(Value(1.0), Neg(Min(Value(0.0), Sub(f_x, flex_cnt))))
                        lty = Min(Value(1.0), Neg(Min(Value(0.0), Sub(flex_cnt, f_y))))
                        remap_x = Min(Max(Div(Sub(flex_cnt, f_x), (Sub(f_y, f_x))), Value(0.0)), Value(1.0))
                        gtey = Neg(Sub(Min(Value(1.0), Neg(Min(Value(0.0), Sub(flex_cnt, f_y)))), Value(1.0)))
                        ltez = Neg(Sub(Min(Value(1.0), Neg(Min(Value(0.0), Sub(f_z, flex_cnt)))), Value(1.0)))
                        gtz = Min(Value(1.0), Neg(Min(Value(0.0), Sub(f_z, flex_cnt))))
                        ltw = Min(Value(1.0), Neg(Min(Value(0.0), Sub(flex_cnt, f_w))))
                        remap_z = Sub(Value(1.0),
                                      Min(Max(Div(Sub(flex_cnt, f_z), (Sub(f_w, f_z))), Value(0.0)), Value(1.0)))
                        final_expr = Add(Add(Mul(Mul(gtx, lty), remap_x), Mul(gtey, ltez)), Mul(Mul(gtz, ltw), remap_z))

                        final_expr = Mul(final_expr, FetchController(self.flex_controllers[op.index].name))
                        stack.append(final_expr)
                    elif flex_op == FlexOpType.DME_UPPER_EYELID:
                        stack.pop(-1)
                        stack.pop(-1)
                        stack.pop(-1)
                        stack.append(Value(1.0))
                    elif flex_op == FlexOpType.DME_LOWER_EYELID:
                        stack.pop(-1)
                        stack.pop(-1)
                        stack.pop(-1)
                        stack.append(Value(1.0))
                    else:
                        print("Unknown OP", op)
                if len(stack) > 1 or not stack:
                    print(f"failed to parse ({self.flex_names[rule.flex_index]}) flex rule")
                    print(stack)
                    continue
                final_expr = stack.pop(-1)
                # name = self.get_value('stereo_flexes').get(rule.flex_index, self.flex_names[rule.flex_index])
                name = self.flex_names[rule.flex_index]
                rules[name] = final_expr
            except Exception as ex:
                traceback.print_exc()
                print(f"failed to parse ({self.flex_names[rule.flex_index]}) flex rule")
                print(stack)

        return rules
