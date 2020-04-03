import traceback
from typing import List

from ...byte_io_mdl import ByteIO
from ..data_structures.source_shared import SourceVertex


class SourceVvdFileData:
    def __init__(self):
        self.id = ""
        self.version = 0
        self.checksum = 0
        self.lod_count = 0
        self.lod_vertex_count = []  # type: List[int]
        self.fixup_count = 0
        self.fixup_table_offset = 0
        self.vertex_data_offset = 0
        self.tangent_data_offset = 0
        self.vertexes_by_lod = {}
        self.fixed_vertexes_by_lod = {}
        self.vertexes = []  # type: List[SourceVertex]
        self.fixups = []  # type: List[SourceVvdFixup]

    def read(self, reader: ByteIO):
        self.id = reader.read_fourcc()
        if self.id != 'IDSV':
            if self.id[:-1] == 'DSV':
                reader.insert_begin(b'I')
                self.id = reader.read_fourcc()
            else:
                raise NotImplementedError(
                    'VVD format {} is not supported!'.format(
                        self.id))
        self.version = reader.read_uint32()
        self.checksum = reader.read_uint32()
        self.lod_count = reader.read_uint32()
        self.lod_vertex_count = [reader.read_uint32() for _ in range(8)]
        self.fixup_count = reader.read_uint32()
        self.fixup_table_offset = reader.read_uint32()
        self.vertex_data_offset = reader.read_uint32()
        self.tangent_data_offset = reader.read_uint32()

        if self.lod_count <= 0:
            return

        reader.seek(self.vertex_data_offset)
        for _ in range(self.lod_vertex_count[0]):
            self.vertexes.append(SourceVertex().read(reader))

        reader.seek(self.fixup_table_offset)
        if self.fixup_count > 0:
            for _ in range(self.fixup_count):
                self.fixups.append(SourceVvdFixup().read(reader))
        # if self.lod_count > 0:
        #     for lod_index in range(self.lod_count):
        #         for fixup in self.fixups:
        #             if fixup.lod_index >= lod_index:
        #                 for j in range(fixup.vertex_count):
        #                     vertex = self.vertexes[fixup.vertex_index + j]
        #                     self.fixed_vertexes_by_lod[lod_index][fixup.vertex_index] = vertex

    def setup_fixed_vertexes(self, lod_index: int):
        self.fixed_vertexes_by_lod[lod_index] = []
        try:
            for fixup_index in range(len(self.fixups)):
                fixup = self.fixups[fixup_index]
                if fixup.lod_index >= lod_index:
                    for j in range(fixup.vertex_count):
                        studio_vertex = self.vertexes[fixup.vertex_index + j]
                        self.fixed_vertexes_by_lod[lod_index].append(studio_vertex)
        except Exception as ex:
            traceback.print_exc()
            print('exception', ex)
            pass

    def __str__(self):
        return "<FileData id:{} version:{} lod count:{} fixup count:{}>".format(self.id, self.version, self.lod_count,
                                                                                self.fixup_count)

    def __repr__(self):
        return self.__str__()


class SourceVvdFixup:
    def __init__(self):
        self.lod_index = 0
        self.vertex_index = 0
        self.vertex_count = 0

    def read(self, reader: ByteIO):
        self.lod_index = reader.read_uint32()
        self.vertex_index = reader.read_uint32()
        self.vertex_count = reader.read_uint32()
        return self

    def __str__(self):
        return "<Fixup lod index:{} vertex index:{} vertex count:{}>".format(self.lod_index, self.vertex_index,
                                                                             self.vertex_count)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    model_path = r"H:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\tf\models\player\demo.mdl"
    SourceVvdFileData()
