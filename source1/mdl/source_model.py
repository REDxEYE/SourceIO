from pathlib import Path

from ...byte_io_mdl import ByteIO
from ..mdl import qc_generator
from ..mdl.mdl_readers.mdl_v10 import SourceMdlFile10
from ..mdl.mdl_readers.mdl_v48 import SourceMdlFile48
from ..mdl.mdl_readers.mdl_v49 import SourceMdlFile49
from ..mdl.mdl_readers.mdl_v53 import SourceMdlFile53
from ..mdl.vtx_readers.vtx_v7 import SourceVtxFile7
from ..mdl.vvd_readers.vvd_v4 import SourceVvdFile4
from ...utilities.path_utilities import case_insensitive_file_resolution


class SourceModel:
    mdl_version_list = {
        10: SourceMdlFile10,
        49: SourceMdlFile49,
        48: SourceMdlFile48,
        44: SourceMdlFile48,
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
            self.mdl_version = self.mdl_version_list[self.version]
        else:
            raise NotImplementedError(
                'Unsupported mdl v{} version'.format(
                    self.version))
        self.vvd = None
        self.vtx = None
        self.mdl:SourceMdlFile49 = None

    def get_mdl_container(self):
        return self.mdl_version(self.mdl_reader)

    def read(self):
        self.mdl = self.mdl_version(self.mdl_reader)
        self.mdl.read()

        version = self.version
        if version == 53:
            self.handle_v53()
        elif version == 10:
            pass
        else:
            self.find_vtx_vvd()
        if self.vvd:
            self.vvd.read()
        if self.vtx:
            self.vtx.read()

    def handle_v53(self):
        self.vvd_reader = self.mdl.vvd  # type: ByteIO
        vvd_magic, vvd_version = self.vvd_reader.peek_fmt('II')
        if vvd_magic != 1448297545:
            raise TypeError('Not a VVD file')

        if vvd_version in self.vvd_version_list:
            self.vvd = self.vvd_version_list[vvd_version](self.vvd_reader)
        else:
            raise NotImplementedError(
                'Unsupported vvd v{} version'.format(vvd_version))

        self.vtx_reader = self.mdl.vtx  # type: ByteIO
        vtx_version = self.vtx_reader.peek_int32()

        if vtx_version in self.vtx_version_list:
            self.vtx = self.vtx_version_list[vtx_version](self.vtx_reader)
        else:
            raise NotImplementedError(
                'Unsupported vtx v{} version'.format(vtx_version))

    def find_vtx_vvd(self):
        vvd = case_insensitive_file_resolution(
            self.filepath.with_suffix('.vvd').absolute())
        vtx = case_insensitive_file_resolution(
            Path(self.filepath.parent / (self.filepath.stem + '.dx90.vtx')).absolute())

        self.vvd_reader = ByteIO(path=vvd)
        vvd_magic, vvd_version = self.vvd_reader.peek_fmt('II')
        if vvd_magic != 1448297545:
            raise TypeError('Not a VVD file')

        if vvd_version in self.vvd_version_list:
            self.vvd = self.vvd_version_list[vvd_version](self.vvd_reader)
        else:
            raise NotImplementedError(
                'Unsupported vvd v{} version'.format(vvd_version))

        self.vtx_reader = ByteIO(path=vtx)
        vtx_version = self.vtx_reader.peek_int32()

        if vtx_version in self.vtx_version_list:
            self.vtx = self.vtx_version_list[vtx_version](self.vtx_reader)
        else:
            raise NotImplementedError(
                'Unsupported vtx v{} version'.format(vtx_version))


if __name__ == '__main__':
    a = SourceModel(
        # r"H:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\Furry\models\male_snake\male_snake.mdl")
        r"F:\PYTHON_STUFF\SourceIO_addon\test_data\V44\bridge_railings001.mdl")
        # r"F:\PYTHON_STUFF\SourceIO_addon\test_data\postal_babe.mdl")
    a.read()

    qc = qc_generator.QC(a)
    qc_path = Path(a.filepath).with_suffix('.qc')
    with qc_path.open('w') as qc_file:
        qc.write_header(qc_file)
        qc.write_models(qc_file)
        qc.write_skins(qc_file)
        qc.write_misc(qc_file)
        qc.write_sequences(qc_file)

    ...
