from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt

from ....shared.types import Vector4
from ....utils import Buffer


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


class FlexControllerRemapType(IntEnum):
    PASSTHRU = 0
    # Control 0 -> ramps from 1-0 from 0->0.5. Control 1 -> ramps from 0-1
    # from 0.5->1
    TWO_WAY = 1
    # StepSize = 1 / (control count-1) Control n -> ramps from 0-1-0 from
    # (n-1)*StepSize to n*StepSize to (n+1)*StepSize. A second control is
    # needed to specify amount to use
    NWAY = 2
    EYELID = 3


@dataclass(slots=True)
class FlexController:
    group: str
    name: str
    local_to_global: int
    min: float
    max: float

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        start_offset = buffer.tell()
        return cls(buffer.read_source1_string(start_offset),
                   buffer.read_source1_string(start_offset),
                   buffer.read_int32(),
                   *(buffer.read_fmt('2f')))


@dataclass(slots=True)
class FlexControllerUI:
    name: str
    controller: Optional[str]
    left_controller: Optional[str]
    right_controller: Optional[str]
    nway_controller: Optional[str]
    remap_type: FlexControllerRemapType
    stereo: bool = False
    unused = []

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        start_offset = buffer.tell()
        name = buffer.read_source1_string(start_offset)
        # TODO: https://github.com/Dmillz89/SourceSDK2013/blob/master/mp/src/public/studio.h#L924
        index0, index1, index2 = buffer.read_fmt('3i')
        remap_type = FlexControllerRemapType(buffer.read_uint8())
        stereo = buffer.read_uint8()
        buffer.skip(2)
        nway_controller = None
        controller = None
        left_controller = None
        right_controller = None
        with buffer.save_current_offset():
            if remap_type == FlexControllerRemapType.NWAY:
                buffer.seek(start_offset + index2)
                buffer.skip(4)
                nway_controller = buffer.read_source1_string(start_offset + index2)
            elif remap_type == FlexControllerRemapType.EYELID:
                buffer.seek(start_offset + index2)
                buffer.skip(4)
                nway_controller = buffer.read_source1_string(start_offset + index2)

            if not stereo:
                buffer.seek(start_offset + index0)
                buffer.skip(4)
                controller = buffer.read_source1_string(start_offset + index0)
            elif stereo:
                buffer.seek(start_offset + index0)
                buffer.skip(4)
                left_controller = buffer.read_source1_string(start_offset + index0)

                buffer.seek(start_offset + index1)
                buffer.skip(4)
                right_controller = buffer.read_source1_string(start_offset + index1)
            else:
                raise RuntimeError('Should never reach this')
        return cls(name, controller, left_controller, right_controller, nway_controller, remap_type, stereo)

    # def __repr__(self):
    #     return f'<FlexControllerUI "{self.name}">'


@dataclass(slots=True)
class FlexOp:
    op: FlexOpType
    value: Union[float, int]

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        op = FlexOpType(buffer.read_uint32())
        if op == FlexOpType.CONST:
            value = buffer.read_float()
        else:
            value = buffer.read_uint32()
        return cls(op, value)

    def __repr__(self):
        return f"FlexOp({self.op.name} {self.value})"


@dataclass(slots=True)
class FlexRule:
    flex_index: int
    flex_ops: List[FlexOp]

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        start_offset = buffer.tell()
        flex_index = buffer.read_uint32()
        op_count = buffer.read_uint32()
        op_offset = buffer.read_uint32()
        flex_ops = []
        if op_count > 0 and op_offset != 0:
            with buffer.read_from_offset(start_offset + op_offset):
                for _ in range(op_count):
                    flex_op = FlexOp.from_buffer(buffer, version)
                    flex_ops.append(flex_op)
        return cls(flex_index, flex_ops)


class VertexAminationType(IntEnum):
    NORMAL = 0
    WRINKLE = 1


@dataclass(slots=True)
class Flex:
    flex_desc_index: int
    targets: Vector4[float]

    partner_index: Optional[int]
    vertex_anim_type: VertexAminationType
    vertex_animations: Optional[npt.NDArray] = field(repr=False)

    def __eq__(self, other: 'Flex'):
        return self.flex_desc_index == other.flex_desc_index and self.targets == other.targets

    def __hash__(self):
        return hash(self.flex_desc_index) + hash(self.targets)

    @classmethod
    def from_buffer(cls, buffer: Buffer, version: int):
        start_offset = buffer.tell()
        flex_desc_index = buffer.read_uint32()

        targets = buffer.read_fmt('4f')
        vert_count, vert_offset = buffer.read_fmt('2I')

        if version > 36:
            partner_index = buffer.read_int32()
            vertex_anim_type = VertexAminationType(buffer.read_uint8())
            if vertex_anim_type == VertexAminationType.WRINKLE:
                vert_anim_class = VertAnimWrinkleV49
            else:
                vert_anim_class = VertAnimV49
            buffer.skip(3+6*4)
        else:
            partner_index = None
            vertex_anim_type = VertexAminationType.NORMAL
            vert_anim_class = VertAnimV36

        if vert_count > 0 and vert_offset != 0:
            with buffer.read_from_offset(start_offset + vert_offset):
                vertex_animations = np.frombuffer(buffer.read(vert_count * vert_anim_class.dtype.itemsize),
                                                  vert_anim_class.dtype)
        else:
            vertex_animations = None

        return cls(flex_desc_index, targets, partner_index, vertex_anim_type, vertex_animations)


class VertAnimV36:
    is_wrinkle = False
    dtype = np.dtype(
        [
            ('index', np.uint32, (1,)),
            ('vertex_delta', np.float32, (3,)),
            ('normal_delta', np.float32, (3,)),
        ]
    )


class VertAnimV49:
    is_wrinkle = False
    dtype = np.dtype(
        [
            ('index', np.uint16, (1,)),
            ('speed', np.uint8, (1,)),
            ('side', np.uint8, (1,)),
            ('vertex_delta', np.float16, (3,)),
            ('normal_delta', np.float16, (3,)),
        ]
    )


class VertAnimWrinkleV49(VertAnimV49):
    is_wrinkle = True
    dtype = np.dtype(
        [
            ('index', np.uint16, (1,)),
            ('speed', np.uint8, (1,)),
            ('side', np.uint8, (1,)),
            ('vertex_delta', np.float16, (3,)),
            ('normal_delta', np.float16, (3,)),
            ('wrinkle_delta', np.float16, (1,)),
        ]
    )
