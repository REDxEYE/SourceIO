import traceback
from dataclasses import dataclass
from typing import List

from ....utils import Buffer
from .. import Mdl
from ..structs.attachment import Attachment
from ..structs.bodygroup import BodyPart
from ..structs.bone import Bone
from ..structs.flex import FlexController, FlexOpType, FlexRule
from ..structs.header import MdlHeaderV36
from ..structs.material import MaterialV36
from ..v49.flex_expressions import *


@dataclass(slots=True)
class MdlV36(Mdl):
    header: MdlHeaderV36

    bones: List[Bone]
    skin_groups: List[List[str]]
    materials: List[MaterialV36]
    materials_paths: List[str]

    flex_names: List[str]
    flex_controllers: List[FlexController]
    flex_rules: List[FlexRule]

    body_parts: List[BodyPart]

    attachments: List[Attachment]

    bone_table_by_name = []

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = MdlHeaderV36.from_buffer(buffer)

        bones = []
        buffer.seek(header.bone_offset)
        for bone_id in range(header.bone_count):
            bone = Bone.from_buffer(buffer, header.version)
            bone.bone_id = bone_id
            bones.append(bone)

        materials = []
        buffer.seek(header.texture_offset)
        for _ in range(header.texture_count):
            texture = MaterialV36.from_buffer(buffer, header.version)
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
                if a != b:
                    diff_start = max(n, diff_start)
                    break

        for n, skin_info in enumerate(skin_groups):
            skin_groups[n] = skin_info[diff_start:]

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

        body_parts = []
        buffer.seek(header.body_part_offset)
        for _ in range(header.body_part_count):
            body_part = BodyPart.from_buffer(buffer, header.version)
            body_parts.append(body_part)

        return cls(header, bones, skin_groups, materials, materials_paths, flex_names, flex_controllers, flex_rules,
                   body_parts, attachments)

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
                        stack.append(FetchController(self.flex_controllers[op.value].name))
                    elif flex_op == FlexOpType.FETCH2:
                        stack.append(FetchFlex(self.flex_names[op.value]))
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
                        count = op.value
                        values = [stack.pop(-1) for _ in range(count)]
                        combo = Combo(*values)
                        stack.append(combo)
                    elif flex_op == FlexOpType.DOMINATE:
                        count = op.value + 1
                        values = [stack.pop(-1) for _ in range(count)]
                        dom = Dominator(*values)
                        stack.append(dom)
                    elif flex_op == FlexOpType.TWO_WAY_0:
                        mx = Max(Add(FetchController(self.flex_controllers[op.value].name), Value(1.0)), Value(0.0))
                        mn = Min(mx, Value(1.0))
                        res = Sub(1, mn)
                        stack.append(res)
                    elif flex_op == FlexOpType.TWO_WAY_1:
                        mx = Max(FetchController(self.flex_controllers[op.value].name), Value(0.0))
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

                        final_expr = Mul(final_expr, FetchController(self.flex_controllers[op.value].name))
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
