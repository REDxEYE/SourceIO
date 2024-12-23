import zipfile
from io import BytesIO

from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager.providers.zip_content_provider import ZIPContentProvider
from SourceIO.library.source1.bsp import Lump, LumpInfo, lump_tag
from SourceIO.library.source1.bsp.bsp_file import BSPFile
from SourceIO.library.utils import Buffer
from SourceIO.library.utils.tiny_path import TinyPath


@lump_tag(40, 'LUMP_PAK')
class PakLump(Lump, ZIPContentProvider):

    def __init__(self, lump_info: LumpInfo):
        super().__init__(lump_info)
        self.filepath = None
        self._steamapp_id = SteamAppId.UNKNOWN
        self._zip_file = None
        self._cache = {}

    def parse(self, buffer: Buffer, bsp: BSPFile):
        self.filepath = bsp.filepath
        if self._zip_file is None:
            zip_data = BytesIO(buffer.read())
            self._zip_file = zipfile.ZipFile(zip_data)
            self._cache = {TinyPath(a.lower()).as_posix(): a for a in self._zip_file.NameToInfo}
        return self
