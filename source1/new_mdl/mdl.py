from typing import List

from ...byte_io_mdl import ByteIO
from ..new_shared.base import Base

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

        class IntermediateExpr:
            def __init__(self, value, precedence):
                self.value = value
                self.precedence = precedence

            def __repr__(self):
                return str(self.value)

        for rule in self.flex_rules:
            stack = []
            try:
                for op in rule.flex_ops:
                    flex_op = op.op
                    if flex_op == FlexOpType.CONST:
                        stack.append(IntermediateExpr(op.value, 10))
                    elif flex_op == FlexOpType.FETCH1:
                        stack.append(IntermediateExpr(self.flex_controllers[op.index].name, 10))
                    elif flex_op == FlexOpType.FETCH2:
                        stack.append(IntermediateExpr(f"%{self.flex_names[op.index]}", 10))
                    elif flex_op == FlexOpType.ADD:
                        right = stack.pop(-1).value
                        left = stack.pop(-1).value
                        stack.append(IntermediateExpr(f"{left} + {right}", 1))
                    elif flex_op == FlexOpType.SUB:
                        right = stack.pop(-1).value
                        left = stack.pop(-1).value
                        stack.append(IntermediateExpr(f"{left} - {right}", 1))
                    elif flex_op == FlexOpType.MUL:
                        right = stack.pop(-1)
                        left = stack.pop(-1)
                        stack.append(
                            IntermediateExpr(f"{left.value if left.precedence > 2 else f'({left.value})'} * "
                                             f"{right.value if right.precedence > 2 else f'({right.value})'}", 2))
                    elif flex_op == FlexOpType.DIV:
                        right = stack.pop(-1)
                        left = stack.pop(-1)
                        stack.append(
                            IntermediateExpr(f"{left.value if left.precedence > 2 else f'({left.value})'} / "
                                             f"{right.value if right.precedence > 2 else f'({right.value})'}", 2))
                    elif flex_op == FlexOpType.NEG:
                        stack.append(IntermediateExpr(f"-{stack.pop(-1).value}", 10))
                    elif flex_op == FlexOpType.MAX:
                        right = stack.pop(-1)
                        left = stack.pop(-1)
                        stack.append(
                            IntermediateExpr(f"max({left.value if left.precedence > 5 else f'({left.value})'}, "
                                             f"{right.value if right.precedence > 5 else f'({right.value})'})", 5))
                    elif flex_op == FlexOpType.MIN:
                        right = stack.pop(-1)
                        left = stack.pop(-1)
                        stack.append(
                            IntermediateExpr(f"min({left.value if left.precedence > 5 else f'({left.value})'}, "
                                             f"{right.value if right.precedence > 5 else f'({right.value})'})", 5))
                    elif flex_op == FlexOpType.COMBO:
                        count = op.index
                        inter = stack.pop(-1)
                        result = str(inter.value)
                        for i in range(2, count):
                            inter = stack.pop(-1)
                            result += f" * {inter.value}"
                        stack.append(IntermediateExpr(f"({result})", 5))
                    elif flex_op == FlexOpType.DOMINATE:
                        count = op.index
                        inter = stack.pop(-1)
                        result = inter.value
                        for i in range(2, count):
                            inter = stack.pop(-1)
                            result += " * " + inter.value
                        inter = stack.pop(-1)
                        result = f"({inter.value} * (1 - {result}))"
                        stack.append(IntermediateExpr(result, 5))
                    else:
                        print("Unknown OP", op)
                if len(stack) > 1 or len(stack) == 0:
                    print(f"failed to parse ({self.flex_names[rule.flex_index]}) flex rule")
                    continue
                final_expr = stack.pop(-1)
                # print(self.flex_names[rule.flex_index], '=', final_expr)
                rules[self.flex_names[rule.flex_index]] = final_expr
            except:
                print(f"failed to parse ({self.flex_names[rule.flex_index]}) flex rule")
                pass

        return rules
