from pathlib import Path

from byte_io_mdl import ByteIO
from mdl_readers.mdl_v49 import SourceMdlFile49
from mdl_readers.mdl_v53 import SourceMdlFile53
from utilities.path_utilities import case_insensitive_file_resolution
from vtx_readers.vtx_v7 import SourceVtxFile7
from vvd_readers.vvd_v4 import SourceVvdFile4


class SourceModel:
    mdl_version_list = {
        49: SourceMdlFile49,
        53: SourceMdlFile53
    }
    vvd_version_list = {
        4: SourceVvdFile4,
    }

    vtx_version_list = {
        7: SourceVtxFile7
    }

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.mdl_reader = ByteIO(path=filepath)
        self.vvd_reader = None
        self.vtx_reader = None
        magic, self.version = self.mdl_reader.peek_fmt('II')
        if self.version in self.mdl_version_list:
            self.mdl = self.mdl_version_list[self.version]
        else:
            raise NotImplementedError('Unsupported mdl v{} version'.format(self.version))
        self.vvd = None
        self.vtx = None

    def read(self):
        self.mdl = self.mdl(self.mdl_reader)
        self.mdl.read()

        version = self.version
        if version == 53:
            self.handle_v53()
        else:
            self.find_vtx_vvd()
        if self.vvd:
            self.vvd.read()
        if self.vtx:
            self.vtx.read()

    def handle_v53(self):
        self.vvd_reader = self.mdl.vvd
        vvd_magic, vvd_version = self.vvd_reader.peek_fmt('II')
        if vvd_magic != 1448297545:
            raise TypeError('Not a VVD file')

        if vvd_version in self.vvd_version_list:
            self.vvd = self.vvd_version_list[vvd_version](self.vvd_reader)
        else:
            raise NotImplementedError('Unsupported vvd v{} version'.format(vvd_version))

        self.vtx_reader = self.mdl.vtx
        vtx_version = self.vtx_reader.peek_int32()

        if vtx_version in self.vtx_version_list:
            self.vtx = self.vtx_version_list[vtx_version](self.vtx_reader)
        else:
            raise NotImplementedError('Unsupported vtx v{} version'.format(vtx_version))

    def find_vtx_vvd(self):
        vvd = case_insensitive_file_resolution(self.filepath.with_suffix('.vvd').absolute())
        vtx = case_insensitive_file_resolution(
            Path(self.filepath.parent / (self.filepath.stem + '.dx90.vtx')).absolute())

        self.vvd_reader = ByteIO(path=vvd)
        vvd_magic, vvd_version = self.vvd_reader.peek_fmt('II')
        if vvd_magic != 1448297545:
            raise TypeError('Not a VVD file')

        if vvd_version in self.vvd_version_list:
            self.vvd = self.vvd_version_list[vvd_version](self.vvd_reader)
        else:
            raise NotImplementedError('Unsupported vvd v{} version'.format(vvd_version))

        self.vtx_reader = ByteIO(path=vtx)
        vtx_version = self.vtx_reader.peek_int32()

        if vtx_version in self.vtx_version_list:
            self.vtx = self.vtx_version_list[vtx_version](self.vtx_reader)
        else:
            raise NotImplementedError('Unsupported vtx v{} version'.format(vtx_version))


if __name__ == '__main__':
    a = SourceModel(r"H:\games\Titanfall 2\extr\models\titans\heavy\sp_titan_heavy_ogre.mdl")
    a.read()
    ...