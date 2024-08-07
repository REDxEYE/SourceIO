import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from SourceIO.library.shared.content_manager.provider import ContentProvider
from ....shared.app_id import SteamAppId
from ....shared.content_manager.providers.zip_content_provider import ZIPContentProvider
from ....utils import Buffer, MemoryBuffer
from .. import Lump, LumpInfo, lump_tag
from ..bsp_file import BSPFile


@lump_tag(40, 'LUMP_PAK')
class PakLump(Lump, ZIPContentProvider):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.filepath = None
        self.zip_file: Optional[zipfile.ZipFile] = None
        self._steamapp_id = SteamAppId.UNKNOWN
        self._zip_file = None
        self._cache = {}

    def parse(self, buffer: Buffer, bsp: 'BSPFile'):
        self.filepath = bsp.filepath
        if self.zip_file is None:
            zip_data = BytesIO(buffer.read())
            self._zip_file = zipfile.ZipFile(zip_data)
            self._cache = {a.lower(): a for a in self._zip_file.NameToInfo}
        return self
