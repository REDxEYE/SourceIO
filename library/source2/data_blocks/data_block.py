from .base_block import DataBlock
from ..utils.binary_keyvalue import BinaryKeyValue
from .ntro_block import NTRO


class DATA(DataBlock):
    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)
        self.data = {}

    @property
    def has_ntro(self):
        return 'NTRO' in [b.block_name for b in self._valve_file.info_blocks]

    def has_ntro_struct(self, struct_name):
        if self.has_ntro:
            ntro_block: NTRO = self._valve_file.get_data_block(block_name="NTRO")[0]
            return ntro_block.get_struct_by_name(struct_name) is not None
        else:
            return False

    @property
    def ntro_struct_name(self):
        if self.has_ntro:
            ntro = self._valve_file.get_data_block(block_name="NTRO")[0]
            return ntro.structs[0].name

    def read(self):
        reader = self.reader
        if reader.size():
            fourcc = reader.peek(4)
            if tuple(fourcc) in BinaryKeyValue.KNOWN_SIGNATURES:
                kv = BinaryKeyValue(self.info_block)
                kv.read(reader)
                self.data = kv.kv
            elif self.has_ntro and self.has_ntro_struct(self.ntro_struct_name):
                ntro = self._valve_file.get_data_block(block_name="NTRO")[0]
                struct_name = {
                    "DATA": ntro.structs[0].name,
                    "PHYS": "VPhysXAggregateData_t",
                }

                struct = ntro.get_struct_by_name(struct_name[self.info_block.block_name])
                self.data = struct.read_struct(reader)
            else:
                raise NotImplementedError(f"Unrecognized DATA block type {fourcc}")
