import sys

from ...data_structures.vtx_data import (SourceVtxFileData,
                                         StripHeaderFlags)
from ....byte_io_mdl import ByteIO


def split(array, n=3):
    return [array[i:i + n] for i in range(0, len(array), n)]


class SourceVtxFile7:
    def __init__(self, reader: ByteIO):
        self.final = False
        self.reader = reader
        self.vtx = SourceVtxFileData()

    def read(self):
        self.vtx.read(self.reader)


if __name__ == '__main__':
    with open('log.log', "w") as f:  # replace filepath & filename
        with f as sys.stdout:
            # model_path = r'G:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\usermod\models\red_eye\tyranno\raptor_subD'
            model_path = r'./test_data\subd'
            # model_path = r'.\test_data\l_pistol_noenv'
            # model_path = r'test_data\geavy'
            # model_path = r'G:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\tf\models\player\heavy'
            # model_path = r'G:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\usermod\models\red_eye\rick-and-morty\pink_raptor'
            # MDL_edit('E:\\MDL_reader\\sexy_bonniev2')
            a = SourceVtxFile7(model_path)
            # a = SourceVtxFile7(r'test_data\kali')
            # a = SourceVtxFile7(r'test_data\kali')

            a.test()
