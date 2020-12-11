from io import BytesIO

from .. import Lump, LumpTypes
import zipfile


class PakLump(Lump):
    lump_id = LumpTypes.LUMP_PAK

    def __init__(self, bsp):
        super().__init__(bsp)
        self.zip_file = None

    def parse(self):
        zip_data = BytesIO(self.reader.read_bytes(self._lump.size))
        self.zip_file = zipfile.ZipFile(zip_data)
        return self
