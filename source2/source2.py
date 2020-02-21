import math
import sys
from pathlib import Path
from pprint import pprint
from typing import List, TextIO, Dict

from typing.io import BinaryIO

from ..byte_io_mdl import ByteIO
import os.path

from .blocks.common import SourceVector
from .blocks.header_block import CompiledHeader, InfoBlock


class ValveFile:

    def __init__(self, filepath):

        from .blocks.ntro_block import NTRO
        from .blocks.redi_block import REDI
        from .blocks.rerl_block import RERL
        from .blocks.vbib_block import VBIB
        from .blocks.data_block import DATA

        print('Reading {}'.format(filepath))
        self.reader = ByteIO(path=filepath, copy_data_from_handle=False, )
        self.filepath = Path(filepath)
        self.filename = self.filepath.name
        self.header = CompiledHeader()
        self.header.read(self.reader)
        self.blocks_info = []  # type: List[InfoBlock]
        self.blocks = {}  # type:Dict[str,InfoBlock]
        self.rerl = RERL(self)
        self.nrto = NTRO(self)
        self.redi = REDI(self)
        self.vbib = VBIB(self)
        self.data = DATA(self)
        self.available_resources = {}

    def read_block_info(self):
        for n in range(self.header.block_count):
            block_info = InfoBlock()
            block_info.read(self.reader)
            self.blocks_info.append(block_info)
            self.blocks[block_info.block_name] = block_info
            print(block_info)
            if block_info.block_name == 'RERL':
                with self.reader.save_current_pos():
                    self.reader.seek(block_info.entry + block_info.block_offset)
                    self.rerl.read(self.reader, block_info)
            elif block_info.block_name == 'NTRO':
                with self.reader.save_current_pos():
                    self.reader.seek(block_info.entry + block_info.block_offset)
                    self.nrto.read(self.reader, block_info)
            elif block_info.block_name == 'REDI':
                with self.reader.save_current_pos():
                    self.reader.seek(block_info.entry + block_info.block_offset)
                    self.redi.read(self.reader, block_info)
            elif block_info.block_name == 'VBIB':
                with self.reader.save_current_pos():
                    self.reader.seek(block_info.entry + block_info.block_offset)
                    self.vbib.read(self.reader, block_info)
            elif block_info.block_name == 'DATA':
                with self.reader.save_current_pos():
                    self.reader.seek(block_info.entry + block_info.block_offset)
                    self.data.read(self.reader, block_info)

    def dump_ntro_block(self, file: BinaryIO):
        for block in self.blocks_info:
            if block.block_name == 'NTRO':
                with self.reader.save_current_pos():
                    self.reader.seek(block.entry + block.block_offset)
                    file.write(self.reader.read_bytes(block.block_size))

    def dump_structs(self, file: TextIO):
        file.write('''struct vector2
{
    float x,y
}
struct vector3
{
    float x,y,z
}
struct vector4
{
    float x,y,z,w
}
struct quaternion
{
    float x,y,z,w
}
struct RGB
{
    byte r,g,b
}
''')
        for struct in self.nrto.structs:
            # print(struct)
            for mem in struct.fields:
                print('\t', mem)
            file.write(struct.as_c_struct())
            # print(struct.as_c_struct())
        for enum in self.nrto.enums:
            # print(enum)
            for mem in enum.fields:
                print('\t', mem)
            file.write(enum.as_c_enum())
            # print(struct.as_c_struct())

    def dump_resources(self):
        for block in self.rerl.resources:
            print(block)
            # for block in self.redi.blocks:
            #     print(block)
            #     for dep in block.container:
            #         print('\t',dep)
            # print(block)
            # for block in self.vbib.vertex_buffer:
            #     for vert in block.vertexes:
            #         print(vert.boneWeight)

    def check_external_resources(self):
        for block in self.rerl.resources:
            path = Path(block.resource_name)
            asset = self.filepath.parent/path.with_suffix(path.suffix + '_c').name
            if asset.exists():
                self.available_resources[block.resource_name] = asset.absolute()
                print('Found', path)
            else:
                print('Can\'t find', path)


def quaternion_to_euler_angle(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    x = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    z = math.degrees(math.atan2(t3, t4))

    return SourceVector(x, y, z)


if __name__ == '__main__':
    with open('log.log', "w") as f:  # replace filepath & filename
        with f as sys.stdout:
            model = r'../test_data/source2/sniper.vmdl_c'
            # model_path = r'../test_data/source2/victory.vanim_c'
            # model_path = r'../test_data/source2/sniper_lod1.vmesh_c'
            # model_path = r'../test_data/source2/sniper_model.vmesh_c'
            # model_path = r'../test_data/source2/gordon_at_desk.vmdl_c'
            # model_path = r'../test_data/source2/abaddon_body.vmat_c'

            # model_path = r'../test_data/source2/sniper_model.vmorf_c'
            # model_path = r'../test_data/source2/sniper.vphys_c'

            vmdl = ValveFile(model)
            vmdl.read_block_info()
            vmdl.dump_structs(open("structures/{}.h".format(model.split('.')[-1]), 'w'))
            vmdl.dump_resources()
            vmdl.check_external_resources()
            # print(vmdl.available_resources)
            model_skeleton = vmdl.data.data['PermModelData_t']['m_modelSkeleton']
            # pprint(model_skeleton)
            bone_names = model_skeleton['m_boneName']
            bone_positions = model_skeleton['m_bonePosParent']
            bone_rotations = model_skeleton['m_boneRotParent']
            bone_parents = model_skeleton['m_nParent']
            for n in range(len(bone_names)):
                print(bone_names[n], 'parent -', bone_names[bone_parents[n]], bone_parents[n], bone_positions[n],
                      quaternion_to_euler_angle(*bone_rotations[n].as_list))
            # print(bone_parents)
            # print(vmdl.available_resources)
            # print(vmdl.header)
