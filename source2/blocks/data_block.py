from .dummy import DataBlock
from ..utils.binary_key_value import BinaryKeyValue


class DATA(DataBlock):
    def __init__(self, valve_file, info_block):
        super().__init__(valve_file, info_block)
        self.data = {}

    def read(self):
        reader = self.reader
        if reader.size():
            with reader.save_current_pos():
                fourcc = reader.read_bytes(4)
            if tuple(fourcc) == (0x56, 0x4B, 0x56, 0x03) or tuple(fourcc) == (0x01, 0x33, 0x56, 0x4B):
                kv = BinaryKeyValue(self.info_block)
                kv.read(reader)
                self.data = kv.kv
            else:
                reader.rewind(4)
                ntro = self._valve_file.get_data_block(block_name="NTRO")[0]
                struct_name = {
                    "DATA": "PermModelData_t" if self._valve_file.filepath.suffix == ".vmdl_c" else "MaterialResourceData_t",
                    "PHYS": "VPhysXAggregateData_t",
                }

                struct = ntro.get_struct_by_name(struct_name[self.info_block.block_name])
                self.data = struct.read_struct(reader)
