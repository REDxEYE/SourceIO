from ...byte_io_mdl import ByteIO
from .dummy import DataBlock
from .header_block import InfoBlock
from .binary_key_value import BinaryKeyValue
from ..source2 import ValveFile


class KV3(DataBlock):
    def __init__(self, valve_file: ValveFile, info_block: InfoBlock):
        super().__init__(valve_file, info_block)
        self.data = {}

    def read(self):
        reader = self.reader
        with reader.save_current_pos():
            fourcc = reader.read_bytes(4)
        if tuple(fourcc) == (0x56, 0x4B, 0x56, 0x03) or tuple(fourcc) == (0x01, 0x33, 0x56, 0x4B):
            kv = BinaryKeyValue(self.info_block)
            kv.read(reader)
            self.data = kv.kv
        else:
            raise NotImplementedError(f"Unknown KV block ({fourcc})")
        self.empty = False
