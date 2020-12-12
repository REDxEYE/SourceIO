from io import BytesIO
from pathlib import Path, PurePath

from .. import Lump, LumpTypes
import zipfile


class PakLump(Lump):
    lump_id = LumpTypes.LUMP_PAK

    def __init__(self, bsp):
        super().__init__(bsp)
        self.zip_file: zipfile.ZipFile = None

    def parse(self):
        zip_data = BytesIO(self.reader.read_bytes(self._lump.size))
        self.zip_file = zipfile.ZipFile(zip_data)
        return self

    def find_file(self, filepath: str, additional_dir=None, extension=None):
        filepath = Path(str(filepath).strip("\\/"))

        new_filepath = filepath
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extension:
            new_filepath = new_filepath.with_suffix(extension)
        if str(new_filepath.as_posix()) in self.zip_file.NameToInfo:
            return self.zip_file.open(str(new_filepath.as_posix()), 'r')
        else:
            return None

    def find_texture(self, filepath):
        return self.find_file(filepath, 'materials', extension='.vtf')

    def find_material(self, filepath):
        return self.find_file(filepath, 'materials', extension='.vmt')
