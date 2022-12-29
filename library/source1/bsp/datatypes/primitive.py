import abc
from typing import TYPE_CHECKING

from ....utils.file_utils import IBuffer

if TYPE_CHECKING:
    from ..lump import Lump
    from ..bsp_file import BSPFile


class Primitive(abc.ABC):
    def __init__(self, lump):
        self._lump: Lump = lump

    def parse(self, reader: IBuffer, bsp: 'BSPFile'):
        raise NotImplementedError()
