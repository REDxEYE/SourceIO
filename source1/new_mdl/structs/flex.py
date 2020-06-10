import struct
from enum import IntEnum

from typing import List, Union

from ....byte_io_mdl import ByteIO
from ...new_shared.base import Base
from .float16 import int16_to_float



class FlexController(Base):
    def __init__(self):
        self.name = ''
        self.type = ''
        self.local_to_global = 0
        self.min = 0.0
        self.max = 0.0

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.type = reader.read_source1_string(entry)
        self.name = reader.read_source1_string(entry)
        self.local_to_global = reader.read_int32()
        self.min, self.max = reader.read_fmt('2f')


class FlexRule(Base):
    def __init__(self):
        self.flex_index = 0
        self.flex_ops = []  # type:List[FlexOp]



    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.flex_index = reader.read_uint32()
        op_count = reader.read_uint32()
        op_offset = reader.read_uint32()
        with reader.save_current_pos():
            if op_count > 0 and op_offset != 0:
                reader.seek(entry + op_offset)
                for _ in range(op_count):
                    flex_op = FlexOp()
                    flex_op.read(reader)
                    self.flex_ops.append(flex_op)


class FlexOpType(IntEnum):
    CONST = 1
    FETCH1 = 2
    FETCH2 = 3
    ADD = 4
    SUB = 5
    MUL = 6
    DIV = 7
    NEG = 8
    EXP = 9
    OPEN = 10
    CLOSE = 11
    COMMA = 12
    MAX = 13
    MIN = 14
    TWO_WAY_0 = 15
    TWO_WAY_1 = 16
    NWAY = 17
    COMBO = 18
    DOMINATE = 19
    DME_LOWER_EYELID = 20
    DME_UPPER_EYELID = 21


class FlexOp(Base):

    def __init__(self):
        self.op = FlexOpType
        self.index = 0
        self.value = 0

    def read(self, reader: ByteIO):
        self.op = FlexOpType(reader.read_uint32())
        if self.op == FlexOpType.CONST:
            self.value = reader.read_float()
        else:
            self.index = reader.read_uint32()

    def __repr__(self):
        return f"FlexOp({self.op.name} {self.index} {self.value})"


class FlexControllerRemapType(IntEnum):
    ASSTHRU = 0
    # Control 0 -> ramps from 1-0 from 0->0.5. Control 1 -> ramps from 0-1
    # from 0.5->1
    TWO_WAY = 1
    # StepSize = 1 / (control count-1) Control n -> ramps from 0-1-0 from
    # (n-1)*StepSize to n*StepSize to (n+1)*StepSize. A second control is
    # needed to specify amount to use
    NWAY = 2
    EYELID = 2


class FlexControllerUI(Base):
    def __init__(self):
        self.name = 0
        self.index1 = 0
        self.index2 = 0
        self.index3 = 0
        self.remap_type = FlexControllerRemapType(0)
        self.stereo = False
        self.unused = []

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.name = reader.read_source1_string(entry)
        #TODO: https://github.com/Dmillz89/SourceSDK2013/blob/master/mp/src/public/studio.h#L924
        self.index1, self.index2, self.index3 = reader.read_fmt('3i')
        self.stereo = reader.read_uint8()
        self.remap_type = FlexControllerRemapType(reader.read_uint8())
        reader.skip(2)


class VertexAminationType(IntEnum):
    NORMAL = 0
    WRINKLE = 1


class Flex(Base):
    def __init__(self):
        self.flex_desc_index = 0
        self.targets = [0.0]

        self.partner_index = 0
        self.vertex_anim_type = 0
        self.vertex_animations = []  # type:List[Union[VertAnimWrinkle,VertAnim]]

    def __eq__(self, other: 'Flex'):
        return self.flex_desc_index == other.flex_desc_index and self.targets == other.targets

    def __hash__(self):
        return hash(self.flex_desc_index) + hash(self.targets)

    def read(self, reader: ByteIO):
        entry = reader.tell()
        self.flex_desc_index = reader.read_uint32()
        self.targets = reader.read_fmt('4f')
        vert_count, vert_offset, self.partner_index = reader.read_fmt(
            '3I')
        self.vertex_anim_type = reader.read_uint8()
        reader.skip(3)
        reader.skip(6 * 4)

        if vert_count > 0 and vert_offset != 0:
            with reader.save_current_pos():
                reader.seek(entry + vert_offset)
                if self.vertex_anim_type == VertexAminationType.WRINKLE:
                    vert_anim_class = VertAnimWrinkle
                else:
                    vert_anim_class = VertAnim
                for _ in range(vert_count):
                    vert_anim = vert_anim_class()
                    vert_anim.read(reader)
                    self.vertex_animations.append(vert_anim)


# noinspection PyBroadException
try:
    struct.calcsize('e')


    def _float16(reader):
        return reader.read_fmt('e')[0]


    def _float16_vector3(reader):
        return reader.read_fmt("3e")
except Exception:
    def _float16(reader):
        return int16_to_float(reader.read_uint16())


    def _float16_vector3(reader):
        return (int16_to_float(reader.read_uint16()),
                int16_to_float(reader.read_uint16()),
                int16_to_float(reader.read_uint16()))


class VertAnim(Base):
    vert_anim_fixed_point_scale = 1 / 4096
    is_wrinkle = False

    def __init__(self):
        self.index = 0
        self.speed = 0
        self.side = 0
        self.vertex_delta = []  # 3
        self.normal_delta = []  # 3

    def read(self, reader: ByteIO):
        self.index, self.speed, self.side = reader.read_fmt('hBB')
        self.vertex_delta = _float16_vector3(reader)
        self.normal_delta = _float16_vector3(reader)
        return self


class VertAnimWrinkle(VertAnim):
    is_wrinkle = True

    def __init__(self):
        super().__init__()
        self.wrinkle_delta = 0

    def read(self, reader: ByteIO):
        super().read(reader)
        self.wrinkle_delta = _float16(reader)
        return self
