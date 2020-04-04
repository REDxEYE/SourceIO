import math
import sys
from pathlib import Path
from pprint import pprint
from typing import List, TextIO, Dict, Tuple
from typing.io import BinaryIO
import os.path

from ..byte_io_mdl import ByteIO
from .blocks.common import SourceVector
from .blocks.header_block import CompiledHeader, InfoBlock
from .blocks.dummy import DataBlock


class ValveFile:

    def __init__(self, filepath):

        # print('Reading {}'.format(filepath))
        self.reader = ByteIO(path=filepath, copy_data_from_handle=False, )
        self.filepath = Path(filepath)
        self.filename = self.filepath.name
        self.header = CompiledHeader()
        self.header.read(self.reader)
        self.info_blocks = []  # type: List[InfoBlock]
        self.data_blocks = []  # type: List[DataBlock]
        self.available_resources = {}

    def read_block_info(self):

        for n in range(self.header.block_count):
            block_info = InfoBlock()
            block_info.read(self.reader)
            self.info_blocks.append(block_info)

        while self.info_blocks:
            block_info = self.info_blocks.pop(0)
            # print(block_info)
            with self.reader.save_current_pos():
                self.reader.seek(block_info.entry + block_info.block_offset)
                block_class = self.get_data_block_class(block_info.block_name)
                if block_class is None:
                    # print(f"Unknown block {block_info}")
                    self.data_blocks.append(None)
                    continue
                block = block_class(self, block_info)
                block.read()
                self.data_blocks.append(block)

    def get_data_block(self, *, block_id=None, block_name=None):
        if block_id is None and block_name is None:
            raise Exception(f"Empty parameters block_id={block_id} block_name={block_name}")
        elif block_id is not None and block_name is not None:
            raise Exception(f"Both parameters filled block_id={block_id} block_name={block_name}")
        if block_id is not None:
            if block_id == -1:
                return None
            return self.data_blocks[block_id]
        if block_name is not None:
            blocks = []
            for block in self.data_blocks:
                if block is not None:
                    if block.info_block.block_name == block_name:
                        blocks.append(block)
            return blocks

    def get_data_block_class(self, block_name):
        from .blocks.ntro_block import NTRO
        from .blocks.redi_block import REDI
        from .blocks.rerl_block import RERL
        from .blocks.vbib_block import VBIB
        from .blocks.data_block import DATA
        from .blocks.kv3_block import KV3
        from .blocks.texture_data_block import TextureData
        if self.filepath.suffix=='.vtex_c':
            data_block_class = TextureData
        else:
            data_block_class = DATA
        data_classes = {
            "NTRO": NTRO,
            "REDI": REDI,
            "RERL": RERL,
            "VBIB": VBIB,
            "DATA": data_block_class,
            "CTRL": KV3,
            "MBUF": VBIB,
            "MDAT": KV3,
            "PHYS": KV3,
            # "ASEQ": KV3,
            # "AGRP": KV3,
            # "ANIM": KV3,
            "MRPH": KV3,
        }
        return data_classes.get(block_name, None)

    def dump_block(self, file: BinaryIO, name: str):
        for block in self.info_blocks:
            if block.block_name == name:
                with self.reader.save_current_pos():
                    self.reader.seek(block.entry + block.block_offset)
                    file.write(self.reader.read_bytes(block.block_size))

    # noinspection PyTypeChecker
    def dump_resources(self):
        from .blocks.rerl_block import RERL
        relr_block: RERL = self.get_data_block(block_name="RERL")[0]
        for block in relr_block.resources:
            print(block)

    def check_external_resources(self):
        relr_block = self.get_data_block(block_name="RERL")[0]
        for block in relr_block.resources:
            path = Path(block.resource_name)
            asset = self.filepath.parent / path.with_suffix(path.suffix + '_c').name
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
