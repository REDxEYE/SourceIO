import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from ....shared.content_providers.content_provider_base import \
    ContentProviderBase
from ....utils import Buffer, MemoryBuffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile


@lump_tag(40, 'LUMP_PAK')
class PakLump(Lump, ContentProviderBase):

    def glob(self, pattern: str):
        raise NotImplementedError

    def find_path(self, filepath: Union[str, Path]):
        pass

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.filepath = None
        self.zip_file: Optional[zipfile.ZipFile] = None
        self._filename_cache = {}

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.filepath = bsp.filepath
        if self.zip_file is None:
            zip_data = BytesIO(buffer.read())
            self.zip_file = zipfile.ZipFile(zip_data)
            self._filename_cache = {a.lower(): a for a in self.zip_file.NameToInfo}
        return self

    def find_file(self, filepath: Union[str, Path], additional_dir=None, extension=None):
        filepath = Path(str(filepath).strip("\\/"))

        new_filepath = filepath
        if additional_dir:
            new_filepath = Path(additional_dir, new_filepath)
        if extension:
            new_filepath = new_filepath.with_suffix(extension)
        new_filepath = str(new_filepath.as_posix()).lower()
        new_filepath = self._filename_cache.get(new_filepath, None)

        if new_filepath is not None:
            return MemoryBuffer(self.zip_file.open(new_filepath, 'r').read())
        return None

    @property
    def steam_id(self):
        return -1
