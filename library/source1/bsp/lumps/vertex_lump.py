import numpy as np
from .. import Lump, lump_tag


@lump_tag(3, 'LUMP_VERTICES')
class VertexLump(Lump):

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.vertices = np.array([])

    def parse(self):
        reader = self.reader
        self.vertices = np.frombuffer(reader.read(), np.float32)
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

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.vertex_info = np.array([])

    def parse(self):
        reader = self.reader
        self.vertex_info = np.frombuffer(reader.read(), self._dtype)
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

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.vertex_info = np.array([])

    def parse(self):
        reader = self.reader
        self.vertex_info = np.frombuffer(reader.read(), self._dtype)
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

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.vertex_info = np.array([])

    def parse(self):
        reader = self.reader
        self.vertex_info = np.frombuffer(reader.read(), self._dtype)
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

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.vertex_info = np.array([])

    def parse(self):
        reader = self.reader
        self.vertex_info = np.frombuffer(reader.read(), self._dtype)
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

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.vertex_info = np.array([])

    def parse(self):
        reader = self.reader
        self.vertex_info = np.frombuffer(reader.read(), self._dtype)
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

    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.vertex_info = np.array([])

    def parse(self):
        reader = self.reader
        self.vertex_info = np.frombuffer(reader.read(), self._dtype)
        return self
