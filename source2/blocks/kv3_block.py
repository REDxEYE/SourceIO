from ...byte_io_mdl import ByteIO
from .dummy import Dummy
from .header_block import InfoBlock
from .binary_key_value import BinaryKeyValue
from ..source2 import ValveFile


class KV3(Dummy):
    def __init__(self, valve_file: ValveFile):
        super().__init__()
        self.valve_file = valve_file
        self.data = {}
        self.info_block = None

    def read(self, reader: ByteIO, block_info: InfoBlock = None):
        self.info_block = block_info
        with reader.save_current_pos():
            fourcc = reader.read_bytes(4)
        if tuple(fourcc) == (0x56, 0x4B, 0x56, 0x03) or tuple(fourcc) == (0x01, 0x33, 0x56, 0x4B):
            kv = BinaryKeyValue(self.info_block)
            kv.read(reader)
            self.data = kv.kv
        else:
            raise NotImplementedError(f"Unknown KV block ({fourcc})")
        self.empty = False
