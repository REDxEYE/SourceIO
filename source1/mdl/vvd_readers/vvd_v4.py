import sys

from ....byte_io_mdl import ByteIO
from ...data_structures.vvd_data import SourceVvdFileData


class SourceVvdFile4:
    def __init__(self, reader: ByteIO):
        self.reader = reader
        self.max_verts = 0
        self.file_data = SourceVvdFileData()

    def read(self):
        self.file_data.read(self.reader)
        self.max_verts = self.file_data.lod_vertex_count[0]
        self.file_data.setup_fixed_vertexes(0)

    def test(self):
        for lod_index,vertexes in self.file_data.fixed_vertexes_by_lod.items():
            for vertex in vertexes:
                print(vertex)


if __name__ == '__main__':
    with open('log.log', "w") as f:  # replace filepath & filename
        with f as sys.stdout:
            # model_path = r'.\test_data\xenomorph'
            # model_path = r'.\test_data\hard_suit'
            # model_path = r'.\test_data\l_pistol_noenv'
            model_path = r"H:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\tf\models\player\demo.vvd"
            reader = ByteIO(path=model_path)
            # MDL_edit('E:\\MDL_reader\\sexy_bonniev2')
            vvd = SourceVvdFile4(reader)
            vvd.read()
            vvd.test()
