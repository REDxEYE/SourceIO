from dataclasses import dataclass, field
from enum import IntFlag
from typing import List, Union

import numpy as np
import numpy.typing as npt

from .....utils import Buffer
from .strip import Strip


# TODO: Verify this against SourceVtxFile107.vb, see line 305 for ref to SourceVtxStripGroup107.*
class StripGroupFlags(IntFlag):
    IS_FLEXED = 0x01
    IS_HWSKINNED = 0x02
    IS_DELTA_FLEXED = 0x04
    SUPPRESS_HW_MORPH = 0x08
    # NOTE: According to Crowbar this constant is made up based on observations in vtx files
    USES_STATIC_PROP_VERTICES = 0x10


VERTEX_DTYPE = np.dtype(
    [
        ('unk1', np.uint16, (1,)),
        # TODO: Is this a bone index rather than a bone weight index?
        ('bone_weight_index', np.uint16, (3,)),
        ('unk2', np.uint16, (1,)),
        ('original_mesh_vertex_index', np.uint16, (1,)),
    ]
)

STATIC_PROP_VERTEX_DTYPE = np.dtype(
    [
        # This is named the same as the one in VERTEX_DTYPE for simplicity when importing
        ('original_mesh_vertex_index', np.uint16, (1,)),
    ]
)


@dataclass(slots=True)
class StripGroup:
    flags: StripGroupFlags
    vertexes: Union[npt.NDArray[VERTEX_DTYPE], npt.NDArray[STATIC_PROP_VERTEX_DTYPE]] = field(repr=False)
    indices: npt.NDArray[np.uint16] = field(repr=False)
    strips: List[Strip] = field(repr=False)

    # topology: List[int]

    @classmethod
    def from_buffer(cls, buffer: Buffer, extra8: bool = False):
        entry = buffer.tell()
        vertex_count = buffer.read_uint16()
        index_count = buffer.read_uint16()
        assert index_count % 3 == 0
        strip_count = buffer.read_uint16()
        flags = StripGroupFlags(buffer.read_uint8())
        unk1 = buffer.read_uint8()
        vertex_offset = buffer.read_uint32()
        index_offset = buffer.read_uint32()
        strip_offset = buffer.read_uint32()
        assert vertex_offset < buffer.size()
        assert strip_offset < buffer.size()
        assert index_offset < buffer.size()
        # TODO: Remove this and the outer loop that uses this when struct.error is thrown
        if extra8:
            buffer.skip(8)
        strips = []
        with buffer.save_current_offset():
            if flags & StripGroupFlags.USES_STATIC_PROP_VERTICES != 0:
                buffer.seek(entry + vertex_offset)
                vertexes = np.frombuffer(buffer.read(vertex_count * STATIC_PROP_VERTEX_DTYPE.itemsize), STATIC_PROP_VERTEX_DTYPE)
            else:
                buffer.seek(entry + vertex_offset)
                vertexes = np.frombuffer(buffer.read(vertex_count * VERTEX_DTYPE.itemsize), VERTEX_DTYPE)
            buffer.seek(entry + index_offset)
            indices = np.frombuffer(buffer.read(index_count * 2), dtype=np.uint16)
            buffer.seek(entry + strip_offset)
            for _ in range(strip_count):
                strip = Strip.from_buffer(buffer, extra8)
                strips.append(strip)

        return cls(flags, vertexes, indices, strips)
