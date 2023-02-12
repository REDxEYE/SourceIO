import abc
from typing import TYPE_CHECKING

from ....utils.file_utils import Buffer

if TYPE_CHECKING:
    from ..bsp_file import BSPFile
    from ..lump import Lump


class Primitive(abc.ABC):
    def __init__(self, lump):
        self._lump: Lump = lump

    def parse(self, reader: Buffer, bsp: 'BSPFile'):
        raise NotImplementedError()
