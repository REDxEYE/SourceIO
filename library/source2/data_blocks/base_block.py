from ...utils.byte_io_mdl import ByteIO


class DataBlock:

    def __init__(self, valve_file, info_block):
        from ..resource_types import ValveCompiledResource
        from .compiled_file_header import InfoBlock

        self._valve_file: ValveCompiledResource = valve_file
        self.info_block: InfoBlock = info_block

        with self._valve_file.reader.save_current_pos():
            self._valve_file.reader.seek(self.info_block.absolute_offset)
            self.reader = ByteIO(self._valve_file.reader.read(self.info_block.block_size))
        self.data = {}
        self.parsed = False

    def read(self):
        self.parsed = True
        raise NotImplementedError()

    def __repr__(self):
        template = '<{} {}>'
        return template.format(type(self).__name__, self.info_block.block_name)
