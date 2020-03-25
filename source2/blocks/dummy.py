from ...byte_io_mdl import ByteIO


class DataBlock:

    def __init__(self, valve_file, info_block):
        from ..source2 import ValveFile
        from .header_block import InfoBlock

        self._valve_file: ValveFile = valve_file
        self.info_block: InfoBlock = info_block
        self.empty = True

    def read(self, reader: ByteIO):
        raise NotImplementedError()

    def __repr__(self):
        template = '<{} {}>'
        return template.format(type(self).__name__, self.info_block.block_name)
