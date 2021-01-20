from io import BytesIO
from pathlib import Path, PurePath

from .. import Lump, lump_tag
import zipfile


@lump_tag(40, 'LUMP_PAK')
class PakLump(Lump):
    def __init__(self, bsp, lump_id):
        super().__init__(bsp, lump_id)
        self.filepath = self._bsp.filepath
        self.zip_file: zipfile.ZipFile = None
        self._cache = {}

    def parse(self):
        zip_data = BytesIO(self.reader.read(self._lump.size))
        self.zip_file = zipfile.ZipFile(zip_data)
        self._cache = {a.lower(): a for a in self.zip_file.NameToInfo}
        return self

    def find_file(self, filepath: str, additional_dir=None, extension=None):
        filepath = Path(str(filepath).strip("\\/"))

        new_filepath = filepath
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extension:
            new_filepath = new_filepath.with_suffix(extension)
        new_filepath = str(new_filepath.as_posix()).lower()
        new_filepath = self._cache.get(new_filepath, None)
        if new_filepath is not None:
            return BytesIO(self.zip_file.open(new_filepath, 'r').read())
        return None
