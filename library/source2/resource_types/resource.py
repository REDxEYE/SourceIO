from pathlib import Path
from typing import List, BinaryIO, Union, Optional, TypeVar, Dict
from ..data_blocks.compiled_file_header import CompiledHeader, InfoBlock
from ..data_blocks import DATA
from ..data_blocks.redi_block import RED2
from ...utils.byte_io_mdl import ByteIO

AnyBlock = TypeVar('AnyBlock', bound='DataBlock')
OptionalBlock = Optional[AnyBlock]


class ValveCompiledResource:
    data_block_class = DATA

    @classmethod
    def parse_new(cls, filepath):
        return cls(filepath)

    def __init__(self, path_or_file):
        self.header = CompiledHeader()
        if path_or_file is not None:
            self.reader = ByteIO(path_or_file)
            self.header.read(self.reader)

        self.info_blocks: List[InfoBlock] = []
        self.data_blocks: List[OptionalBlock] = []
        self.available_resources: Dict[Union[str, int], Path] = {}

        self.read_block_info()
        self.check_external_resources()

    def read_block_info(self):
        self.info_blocks.clear()
        self.data_blocks.clear()
        if self.header.block_count:
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
                with self.reader.save_current_pos():
                    self.reader.seek(block_info.entry + block_info.block_offset)
                    block_class = self.get_data_block_class(block_info.block_name)
                    if block_class is None:
                        self.data_blocks[self.info_blocks.index(block_info)] = None
                        continue
                    block = block_class(self, block_info)
                    self.data_blocks[self.info_blocks.index(block_info)] = block

    def get_data_block(self, *,
                       block_id: Optional[int] = None,
                       block_name: Optional[str] = None) -> Union[OptionalBlock, List[OptionalBlock]]:
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
        from ..data_blocks import DATA, NTRO, REDI, RERL, VBIB, MRPH, ANIM

        data_classes = {
            "NTRO": NTRO,
            "REDI": REDI,
            "RED2": RED2,
            "RERL": RERL,
            "VBIB": VBIB,
            "VXVS": VBIB,
            "DATA": self.data_block_class,
            "CTRL": DATA,
            "MBUF": VBIB,
            "MDAT": DATA,
            "PHYS": DATA,
            "ASEQ": DATA,
            "AGRP": DATA,
            "ANIM": ANIM,
            "MRPH": MRPH,
        }
        return data_classes.get(block_name, None)

    def dump_block(self, file: BinaryIO, name: str):
        for block in self.info_blocks:
            if block.block_name == name:
                with self.reader.save_current_pos():
                    self.reader.seek(block.entry + block.block_offset)
                    file.write(self.reader.read(block.block_size))

    def dump_resources(self):
        from ..data_blocks import RERL
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

    def check_external_resources(self):
        from ..data_blocks import RERL
        if self.get_data_block(block_name="RERL"):
            relr_block: RERL = self.get_data_block(block_name="RERL")[0]
            for block in relr_block.resources:
                path = Path(block.resource_name)
                asset = path.with_suffix(path.suffix + '_c')
                if asset:
                    self.available_resources[block.resource_name] = asset
                    self.available_resources[block.resource_hash] = asset

    def get_child_resource(self, name):
        if self.available_resources.get(name, None) is not None:
            res = ValveCompiledResource(self.available_resources.get(name))
            res.read_block_info()
            res.check_external_resources()
            return res
        return None
