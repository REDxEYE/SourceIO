from ...byte_io_mdl import ByteIO


class DataBlock:

    def __init__(self, valve_file, info_block):
        from ..source2 import ValveFile
        from .header_block import InfoBlock

        self._valve_file: ValveFile = valve_file
        self.info_block: InfoBlock = info_block
        with self._valve_file.reader.save_current_pos():
            self._valve_file.reader.seek(self.info_block.absolute_offset)
            self.reader = ByteIO(byte_object=self._valve_file.reader.read_bytes(self.info_block.block_size))
        self.empty = True

    def read(self):
        raise NotImplementedError()

    def __repr__(self):
        template = '<{} {}>'
        return template.format(type(self).__name__, self.info_block.block_name)
