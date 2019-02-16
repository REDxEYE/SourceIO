import sys


from SourceIO.byte_io_mdl import ByteIO
from SourceIO.data_structures.vvd_data import SourceVvdFileData


class SourceVvdFile4:
    def __init__(self, reader: ByteIO):
        self.reader = reader
        self.max_verts = 0
        self.file_data = SourceVvdFileData()

    def read(self):
        self.file_data.read(self.reader)
        self.max_verts = self.file_data.lod_vertex_count[0]

    def test(self):
        for v in self.file_data.vertexes:
            print(v)


if __name__ == '__main__':
    with open('log.log', "w") as f:  # replace filepath & filename
        with f as sys.stdout:
            # model_path = r'.\test_data\xenomorph'
            # model_path = r'.\test_data\hard_suit'
            # model_path = r'.\test_data\l_pistol_noenv'
            model_path = r'G:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\usermod\models\red_eye\tyranno\raptor'
            # MDL_edit('E:\\MDL_reader\\sexy_bonniev2')
            SourceVvdFile4(model_path).test()
