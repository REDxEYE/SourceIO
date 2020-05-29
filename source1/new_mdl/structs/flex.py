from enum import IntFlag, IntEnum

import numpy as np

from ....byte_io_mdl import ByteIO
from ..base import Base

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
        self.flex_ops = []

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
        self.index1, self.index2, self.index3 = reader.read_fmt('3i')
        self.stereo = reader.read_uint8()
        self.remap_type = FlexControllerRemapType(reader.read_uint8())
        reader.skip(2)
