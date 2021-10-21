
class Primitive:
    def __init__(self, lump, bsp):
        from ..lump import Lump
        from ..bsp_file import BSPFile
        self._lump: Lump = lump
        self._bsp: BSPFile = bsp
