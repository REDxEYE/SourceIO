import numpy as np

from ....utils import Buffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile


@lump_tag(3, 'LUMP_VERTICES')
class VertexLump(Lump):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.vertices = np.array([])

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.vertices = np.frombuffer(buffer.read(), np.float32)
        self.vertices = self.vertices.reshape((-1, 3))
        return self


@lump_tag(0x47, 'LUMP_UNLITVERTEX', bsp_version=29)
class UnLitVertexLump(Lump):
    _dtype = np.dtype(
        [
            ('vpi', np.uint32, (1,)),
            ('vni', np.uint32, (1,)),
            ('uv', np.float32, (2,)),
            ('unk', np.int32, (1,)),
        ]
    )

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.vertex_info = np.array([])

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.vertex_info = np.frombuffer(buffer.read(), self._dtype)
        return self


@lump_tag(0x48, 'LUMP_LITVERTEXFLAT', bsp_version=29)
class LitVertexFlatLump(Lump):
    _dtype = np.dtype(
        [
            ('vpi', np.uint32, (1,)),
            ('vni', np.uint32, (1,)),
            ('uv', np.float32, (2,)),
            ('unk', np.int32, (5,)),
        ]
    )

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.vertex_info = np.array([])

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.vertex_info = np.frombuffer(buffer.read(), self._dtype)
        return self


@lump_tag(0x49, 'LUMP_BUMPLITVERTEX', bsp_version=29)
class BumpLitVertexLump(Lump):
    _dtype = np.dtype(
        [
            ('vpi', np.uint32, (1,)),
            ('vni', np.uint32, (1,)),
            ('uv', np.float32, (2,)),
            ('unk1', np.int32, (1,)),
            ('uv_lm', np.float32, (2,)),
            ('uv1', np.float32, (2,)),
            ('unk2', np.uint32, (2,)),
        ]
    )

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.vertex_info = np.array([])

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.vertex_info = np.frombuffer(buffer.read(), self._dtype)
        return self


@lump_tag(0x4a, 'LUMP_UNLITTSVERTEX', bsp_version=29)
class UnlitTSVertexLump(Lump):
    _dtype = np.dtype(
        [
            ('vpi', np.uint32, (3,)),
            ('vni', np.uint32, (1,)),
            ('uv', np.float32, (2,)),
        ]
    )

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.vertex_info = np.array([])

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.vertex_info = np.frombuffer(buffer.read(), self._dtype)
        return self


@lump_tag(0x4B, 'LUMP_BLINNPHONGVERTEX', bsp_version=29)
class BlinnPhongVertexLump(Lump):
    _dtype = np.dtype(
        [
            ('vpi', np.uint32, (3,)),
            ('vni', np.uint32, (1,)),
            ('unk', np.uint32, (2,)),
        ]
    )

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.vertex_info = np.array([])

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.vertex_info = np.frombuffer(buffer.read(), self._dtype)
        return self


@lump_tag(0x4C, 'LUMP_R5VERTEX', bsp_version=29)
class R5VertexLump(Lump):
    _dtype = np.dtype(
        [
            ('vpi', np.uint32, (3,)),
            ('vni', np.uint32, (1,)),
            ('unk', np.uint32, (2,)),
            ('uv', np.float32, (2,)),
            ('uv_lm', np.float32, (2,)),
        ]
    )

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.vertex_info = np.array([])

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.vertex_info = np.frombuffer(buffer.read(), self._dtype)
        return self


@lump_tag(0x4E, 'LUMP_R7VERTEX', bsp_version=29)
class R7VertexLump(Lump):
    _dtype = np.dtype(
        [
            ('vpi', np.uint32, (3,)),
            ('vni', np.uint32, (1,)),
            ('uv', np.float32, (2,)),
            ('neg_one', np.int32, (1,)),
        ]
    )

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.vertex_info = np.array([])

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.vertex_info = np.frombuffer(buffer.read(), self._dtype)
        return self
