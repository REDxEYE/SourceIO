import math
import sys
from pathlib import Path
from typing import List, BinaryIO, Union

from ..byte_io_mdl import ByteIO
from .common import SourceVector
from .compiled_file_header import CompiledHeader, InfoBlock
from ..utilities.path_utilities import backwalk_file_resolver
from .blocks import DataBlock


class ValveFile:

    @classmethod
    def parse_new(cls, filepath):
        return cls(filepath)

    def __init__(self, filepath, data_block_handler=None):

        # print('Reading {}'.format(filepath))
        self.reader = ByteIO(path=filepath, copy_data_from_handle=False)
        self.filepath = Path(filepath)
        self.data_block_handler = data_block_handler
        self.filename = self.filepath.name
        self.header = CompiledHeader()
        self.header.read(self.reader)
        self.info_blocks = []  # type: List[InfoBlock]
        self.data_blocks = []  # type: List[Union[DataBlock,None]]
        self.available_resources = {}

    def read_block_info(self):
        self.info_blocks.clear()
        self.data_blocks.clear()
        self.reader.seek(4 * 4)
        for n in range(self.header.block_count):
            block_info = InfoBlock()
            block_info.read(self.reader)
            self.info_blocks.append(block_info)
            self.data_blocks.append(None)

        for i in range(len(self.info_blocks)):
            block_info = self.info_blocks[i]
            if self.data_blocks[i] is not None:
                continue
            # print(block_info)
            with self.reader.save_current_pos():
                self.reader.seek(block_info.entry + block_info.block_offset)
                block_class = self.get_data_block_class(block_info.block_name)
                if block_class is None:
                    self.data_blocks[self.info_blocks.index(block_info)] = None
                    continue
                block = block_class(self, block_info)
                self.data_blocks[self.info_blocks.index(block_info)] = block

    def get_data_block(self, *, block_id=None, block_name=None):
        if block_id is None and block_name is None:
            raise Exception(f"Empty parameters block_id={block_id} block_name={block_name}")
        elif block_id is not None and block_name is not None:
            raise Exception(f"Both parameters filled block_id={block_id} block_name={block_name}")
        if block_id is not None:
            if block_id == -1:
                return None
            block = self.data_blocks[block_id]
            if not block.parsed:
                block.reader.seek(0)
                block.read()
                block.parsed = True
            return block
        if block_name is not None:
            blocks = []
            for block in self.data_blocks:
                if block is not None:
                    if block.info_block.block_name == block_name:
                        if not block.parsed:
                            block.read()
                            block.parsed = True
                        blocks.append(block)
            return blocks

    def get_data_block_class(self, block_name):
        from .blocks import TextureBlock, DATA, NTRO, REDI, RERL, VBIB, MRPH
        if self.filepath.suffix == '.vtex_c':
            data_block_class = TextureBlock
        elif self.filepath.suffix == '.vmorf_c':
            data_block_class = MRPH
        else:
            data_block_class = DATA
        data_classes = {
            "NTRO": NTRO,
            "REDI": REDI,
            "RERL": RERL,
            "VBIB": VBIB,
            "VXVS": VBIB,
            "DATA": data_block_class,
            "CTRL": DATA,
            "MBUF": VBIB,
            "MDAT": DATA,
            "PHYS": DATA,
            "ASEQ": DATA,
            "AGRP": DATA,
            "ANIM": DATA,
            "MRPH": MRPH,
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

    @staticmethod
    def get_base_dir(full_path: Path, relative_path: Path):
        addon_path = Path(full_path)
        for p1, p2 in zip(reversed(addon_path.parts), reversed(relative_path.parts)):
            if p1 == p2:
                addon_path = addon_path.parent
            else:
                break
        return addon_path

    # noinspection PyTypeChecker
    def check_external_resources(self):
        from .blocks.rerl_block import RERL
        if self.get_data_block(block_name="RERL"):
            relr_block: RERL = self.get_data_block(block_name="RERL")[0]

            for block in relr_block.resources:
                path = Path(block.resource_name)
                asset = path.with_suffix(path.suffix + '_c')
                asset = backwalk_file_resolver(Path(self.filepath).parent, asset)
                if asset and asset.exists():
                    self.available_resources[block.resource_name] = asset.absolute()
                    # print('Found', path)
                else:
                    pass
                    # print('Can\'t find', path)

    def get_child_resource(self, name):
        if self.available_resources.get(name, None) is not None:
            res = ValveFile(self.available_resources.get(name))
            res.read_block_info()
            res.check_external_resources()
            return res
        return None


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
